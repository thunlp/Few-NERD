import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        del outputs
        word_embeddings = torch.sum(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        return word_embeddings
    
    def tokenize(self, raw_tokens):
        raw_tokens = [token.lower() for token in raw_tokens]
        raw_tokens_list = []
        # split if too long
        while len(raw_tokens) > self.max_length - 2:
            raw_tokens_list.append(raw_tokens[:self.max_length-2])
            raw_tokens = raw_tokens[self.max_length-2:]
        if raw_tokens:
            raw_tokens_list.append(raw_tokens)

        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for raw_tokens in raw_tokens_list:
            # token -> index
            tokens = ['[CLS]'] + raw_tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)
        return indexed_tokens_list, mask_list, text_mask_list
