import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        return word_embeddings
