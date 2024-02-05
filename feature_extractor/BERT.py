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


class BERT_FEATURE_EXTRACTOR(nn.Module):
    def __init__(self,args):
        super(BERT_FEATURE_EXTRACTOR, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.LLM_path)
        self.bert = BertModel.from_pretrained(args.LLM_path, return_dict=True)
        self.max_len = 128  # !
        self.device = args.device

    def forward(self, texts):
        inputs = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True,
                                return_tensors="pt")  # "pt"表示"pytorch"
        inputs = inputs.to(self.device)
        outputs = self.bert(**inputs)[1]
        return outputs

    def save_model(self, model_file_path):
        self.bert.save_pretrained(model_file_path+"/bert-base-chinese")  # 保存微调后的模型
        self.tokenizer.save_pretrained(model_file_path+"/bert-base-chinese")  # 保存tokenizer，以备后续使用

    def load_model(self, model_file_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_file_path+"/bert-base-chinese")
        self.bert = BertModel.from_pretrained(model_file_path+"/bert-base-chinese", return_dict=True)