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
    def __init__(self, LLM_path):
        super(BERT, self).__init__()
        self.LLM_path=LLM_path
        self.tokenizer = BertTokenizer.from_pretrained(self.LLM_path)
        self.bert = BertModel.from_pretrained(self.LLM_path, return_dict=True)
        self.max_len = 128  # !

    def forward(self, texts):
        inputs = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True,
                                return_tensors="pt")  # "pt"表示"pytorch"
        inputs=inputs.to('cuda')
        outputs = self.bert(**inputs)[1]
        return outputs
    
    def save_model(self,epoch_num):
        self.bert.save_pretrained(f"checkpoint/base-LLMs/bert-base-chinese/{epoch_num}")  # 保存微调后的模型
        self.tokenizer.save_pretrained(f"checkpoint/base-LLMs/bert-base-chinese/{epoch_num}")  # 保存tokenizer，以备后续使用

