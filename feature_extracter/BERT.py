from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from utils import Datasets
from torch.optim import Adam
from datetime import datetime
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch
import os
import sys



class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained()
        self.bert = BertModel.from_pretrained(return_dict=True)
        self.max_len=128

    
    def forward(self,texts):
        inputs = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")  # "pt"表示"pytorch"
        outputs = self.bert(**inputs)
        return outputs








