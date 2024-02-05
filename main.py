import argparse
import numpy as np
from utils import get_loader_and_length
from model import UDAModel
import torch.optim as optim
import torch

epoch_num = 1
batch_size = 10
device = 'cuda'
lr = 1e-5
load_epoch = None

log_batch_num = 10

source_path = 'data/civil_sub_test.json'
target_path = 'data/civil_sub_test.json'
test_path = 'data/civil_sub_test.json'

LLM_path = 'Base-LLMs/bert-base-chinese'


def train():
    '''
    定义训练过程：
    
    
    '''
    source_loader, source_len = get_loader_and_length(source_path, batch_size=batch_size)
    target_loader, target_len = get_loader_and_length(target_path, batch_size=batch_size)

    model = UDAModel(LLM_path, load_epoch)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if device == 'cuda':
        model = model.cuda()
        loss_domain = loss_domain.cuda()
        loss_class = loss_class.cuda()

    for epoch in range(epoch_num):
        cur_epoch = 0
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
            domain_labels = torch.zeros(len(class_labels)).long()

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
            domain_labels = torch.ones(len(class_labels)).long()

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
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                      % (epoch, cnt, loader_len, err_s_label.cpu().data.numpy(), err_s_domain.cpu().data.numpy(),
                         err_t_domain.cpu().data.numpy()))
        model.save(load_epoch+epoch_num)
        test(test_path)

        cur_epoch += 1

    print('done')


def test(test_path):
    test_loader, test_len = get_loader_and_length(test_path, batch_size=batch_size)

    model = UDAModel(LLM_path=LLM_path)
    acc_num = 0
    if device == 'cuda':
        model = model.cuda()

    test_iter = iter(test_loader)
    with torch.inference_mode():
        for test_batch in test_iter:
            texts, class_labels = test_batch
            if device == 'cuda':
                class_labels = class_labels.cuda()

            class_output, _ = model(input=texts, trade_off_param=0)
            pred = torch.argmax(class_output, dim=-1)
            acc_num += (pred.long() == class_labels.long()).float().sum()

        acc_num = int(acc_num)
        round_acc = round(acc_num / test_len, 4)
        # acc_num += (output.long() == batch["labels"].long()).float().sum()

    print(f'test acc:{round_acc} correct/total: {acc_num}/{test_len}\n')


if __name__ == '__main__':
    train()
    test(source_path)
