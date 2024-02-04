import argparse
import numpy as np
from utils import get_loader
from model import UDAModel
import torch.optim as optim
import torch

epoch_num = 10
batch_size = 10
device = 'cuda'

log_batch_num = 100

source_path = 'data/civil_train.json'
target_path = 'data/criminal_train.json'
test_path = 'data/criminal_test.json'

LLM_path = 'Base-LLMs/bert-base-chinese'

source_loader = get_loader(source_path, batch_size=batch_size)
target_loader = get_loader(target_path, batch_size=batch_size)
test_loader = get_loader(test_path, batch_size=batch_size)


def train():
    '''
    定义训练过程：
    
    
    '''

    model = UDAModel(LLM_path)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if device == 'cuda':
        model = model.cuda()
        loss_domain = loss_domain.cuda()
        loss_class = loss_class.cuda()

    for epoch in range(epoch_num):
        loader_len = min(len(source_loader), len(target_loader))

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        cnt = 0
        while cnt < loader_len:
            p = float(cnt + epoch * loader_len) / epoch_num / loader_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            '''
            source
            '''
            source_batch = next(source_iter)
            texts, class_labels = source_batch
            class_labels = torch.tensor(class_labels)
            domain_labels = torch.zeros(batch_size).long()

            if device == 'cuda':
                class_labels = class_labels.cuda()
                domain_labels = domain_labels.cuda()

            class_output, domain_output = model(input=texts, trade_off_param=alpha)
            err_s_label = loss_class(class_output, class_labels)
            err_s_domain = loss_domain(domain_output, domain_labels)

            '''
            target
            '''
            target_batch = next(target_iter)
            texts, class_labels = target_batch
            domain_labels = torch.ones(batch_size).long()

            if device == 'cuda':
                domain_labels = domain_labels.cuda()

            _, domain_output = model(input=texts, trade_off_param=alpha)
            err_t_domain = loss_domain(domain_output, domain_labels)

            '''
            total loss and backward
            '''
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            '''
            log
            '''
            cnt += 1
            if cnt % log_batch_num == 0:
                print('log')


def test():
    model = UDAModel()
    acc_num = 0
    with torch.inference_mode():
        for batch in test_loader:
            if device == 'cuda':
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            acc_num += (output.long() == batch["labels"].long()).float().sum()

        print(f'acc:{acc_num / len(test_loader)}\n')


if __name__ == '__main__':
    train()
