import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoModelForQuestionAnswering

## MODEL
class SimpleBertQA(AutoModelForQuestionAnswering): # BertPreTrainedModel

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = AutoModel(config, add_pooling_layer=False) # BertModel
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) # + outputs[2:]
        
        # return ((total_loss,) + output)
        return total_loss, start_logits, end_logits