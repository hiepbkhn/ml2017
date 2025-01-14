import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import CrossEntropyLoss
from einops import rearrange

# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# blocks
def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Linear(dim * mult, dim)
    )

class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting keys to key attention logits

        self.to_r = nn.Linear(dim_head, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        mask_value = -torch.finfo(x.dtype).max
        mask = rearrange(mask, 'b n -> b () n')

        # calculate query attention logits
        q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = q_attn_logits.softmax(dim = -1)

        # calculate global query token
        global_q = einsum('b h n, b h n d -> b h d', q_attn, q)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # now calculate key attention logits
        k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = k_attn_logits.softmax(dim = -1)

        # calculate global key token
        global_k = einsum('b h n, b h n d -> b h d', k_attn, k)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values
        v = v * global_k
        r = self.to_r(v)

        r = r + q # paper says to add the queries as a residual

        # aggregate
        r = rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)

# main class

class FastTransformerBERT(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4
    ):
        super().__init__()
        self.num_tokens = num_tokens # hiepnh
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads)
            ff = FeedForward(dim, mult = ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers
        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        n, device = input_ids.shape[1], input_ids.device
        x = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        bool_mask = attention_mask > 0 # attention_mask.bool()
        for attn, ff in self.layers:
            x = attn(x, mask = bool_mask) + x
            x = ff(x) + x

        # hiepnh
        prediction_scores = self.to_logits(x)
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.num_tokens), labels.view(-1))
        
#         return self.to_logits(x)
        return masked_lm_loss, x
