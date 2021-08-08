import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

MAX_LEN = train_vectors[0].shape[0] # 937 -> 900

class WordEmbedQA(nn.Module):
    def __init__(self):
        super().__init__() #
        
        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 100)
        self.qa_outputs = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        
    def forward(self, x, start_positions, end_positions): #, ignored_index):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.qa_outputs(x))
        logits = x # self.dropout(x)
        
#         print('logits.size =', logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
#         ignored_index = start_logits.size(1)
#         start_positions.clamp_(0, ignored_index)
#         end_positions.clamp_(0, ignored_index)

#         loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits
    
#### LSTM
class WordEmbedLSTMQA(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__() # WordEmbedLSTMQA, self
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(300, self.hidden_dim, batch_first=True) # bidirectional=True, 
        self.qa_outputs = nn.Linear(self.hidden_dim, 2)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        
    def forward(self, x, start_positions, end_positions): #, ignored_index):
        x, _ = self.lstm(x)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = F.relu(self.qa_outputs(x))
        x = x.view(start_positions.size(0), MAX_LEN, -1)
        logits = x 
        
#         print('logits.size =', logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        
        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits

#### BiLSTM   
class WordEmbedBiLSTMQA(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__() # WordEmbedLSTMQA, self
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(300, self.hidden_dim, bidirectional=True, batch_first=True) # bidirectional=True, 
        self.qa_outputs = nn.Linear(2*self.hidden_dim, 2)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        
    def forward(self, x, start_positions, end_positions): #, ignored_index):
        x, _ = self.lstm(x)
        x = x.contiguous().view(-1, 2*self.hidden_dim)
        x = F.relu(self.qa_outputs(x))
        x = x.view(start_positions.size(0), MAX_LEN, -1)
        logits = x 
        
#         print('logits.size =', logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        
        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits    

#### Transformer    
class WordEmbedTransformerQA(nn.Module):
    def __init__(self):
        super().__init__() #
        
        self.encoder = nn.TransformerEncoderLayer(d_model=300, nhead=6)
        self.qa_outputs = nn.Linear(300, 2)
        
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        
    def forward(self, x, start_positions, end_positions): #, ignored_index):
        x = self.encoder(x)
        x = F.relu(self.qa_outputs(x))
        logits = x # self.dropout(x)
        
#         print('logits.size =', logits.size())
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
#         ignored_index = start_logits.size(1)
#         start_positions.clamp_(0, ignored_index)
#         end_positions.clamp_(0, ignored_index)

#         loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits   
    
#### UNet-1D
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1) # add padding
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(300,64,128)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            x        = torch.cat([x, encoder_features[i]], dim=1) 
            x        = self.dec_blocks[i](x)
        return x

class WordEmbedUNetQA(nn.Module):
    def __init__(self, enc_chs=(300,64,128), dec_chs=(128, 64), num_class=2):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv1d(dec_chs[-1], num_class, 1) # kernel-size = 1

    def forward(self, x, start_positions, end_positions):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        logits   = self.head(out)
        logits = torch.transpose(logits, 1, 2)
#         print('logits.size =', logits.size())
        
        start_logits, end_logits = logits.split(1, dim=-1) 
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        loss_fct = CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits   