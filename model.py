import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertForQuestionAnswering
from modules import MultiNonLinearClassifier

class BertQueryNerConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertQueryNerConfig, self).__init__(**kwargs)
        self.mrc_dropout = kwargs.get("mrc_dropout", 0.1)
    

class BERTMRC(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTMRC, self).__init__(config)
        self.bert = BertModel(config)
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)

        self.hidden_size = config.hidden_size

        self.init_weights()

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        # print("start_logits: {}".format(start_logits.size()))
        # print("end_logits: {}".format(end_logits.size()))

        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


class BERTPretrainedMRC(nn.Module):
    def __init__(self, bert_dir, args):
        super(BERTPretrainedMRC, self).__init__()
        if args.load_pretrainedBERT:
            self.bert = BertForQuestionAnswering.from_pretrained(bert_dir)
        else:
            self.bert_config = BertQueryNerConfig.from_pretrained(bert_dir,
                                                                  hidden_dropout_prob=args.bert_dropout,
                                                                  attention_probs_dropout_prob=args.bert_dropout,
                                                                  mrc_dropout=args.mrc_dropout)
            self.bert = BertForQuestionAnswering(self.bert_config)
        # self.start_outputs = nn.Linear(config.hidden_size, 1)
        # self.end_outputs = nn.Linear(config.hidden_size, 1)
        # self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)

        # self.hidden_size = config.hidden_size

        # self.init_weights()

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        start_logits = bert_outputs.start_logits
        end_logits = bert_outputs.end_logits
        
        return start_logits, end_logits, None


