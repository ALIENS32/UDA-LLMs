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
    def __init__(self, args):
        super(BERT_FEATURE_EXTRACTOR, self).__init__()
        # 加载模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(args.LLM_path)
        self.bert = BertModel.from_pretrained(args.LLM_path, return_dict=True)

        self.max_len = 128  # ! 这个参数可能需要设为超参数，代表一句话最大长度，超过要截断，不足要填充，tokenizer的参数
        self.device = args.device

    def forward(self, texts):
        inputs = self.tokenizer(texts, max_length=self.max_len, padding="max_length", truncation=True,
                                return_tensors="pt")  # "pt"表示"pytorch"
        inputs = inputs.to(self.device)
        outputs = self.bert(**inputs)[1]  # 输出的第二个维度是特征
        return outputs

    def save_model(self, model_file_path):
        self.bert.save_pretrained(model_file_path + "/bert-base-chinese")  # 保存微调后的模型
        self.tokenizer.save_pretrained(model_file_path + "/bert-base-chinese")  # 保存tokenizer，以备后续使用

    def load_model(self, model_file_path):
        # 从checkpoint下指定的模型文件夹加载
        self.tokenizer = BertTokenizer.from_pretrained(model_file_path + "/bert-base-chinese")
        self.bert = BertModel.from_pretrained(model_file_path + "/bert-base-chinese", return_dict=True)
