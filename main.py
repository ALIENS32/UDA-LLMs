import argparse
import numpy as np
from utils import get_loader
from model import UDAModel
import torch.optim as optim
import torch


epoch_num=0
batch_size=10
device='cuda'

log_batch_num=100


source_loader=get_loader()
target_loader=get_loader()
test_loader=get_loader()

def train():
    '''
    定义训练过程：
    
    
    '''
    
    model=UDAModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()


    
    for epoch in range(epoch_num):
        loader_len=min(len(source_loader),len(target_loader))


        
        cnt=0
        while cnt<loader_len:
            p = float(cnt + epoch * loader_len) / epoch_num / loader_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            '''
            source
            '''
            source_batch=next(source_loader)
            texts,class_labels=source_batch
            domain_labels = torch.zeros(batch_size)

            if device=='cuda':
                texts=texts.cuda()
                class_labels=class_labels.cuda()
                domain_labels=domain_labels.cuda()

            class_output, domain_output = model(input_data=texts, alpha=alpha)
            err_s_label = loss_class(class_output, class_labels)
            err_s_domain = loss_domain(domain_output, domain_labels)


            '''
            target
            '''
            target_batch=next(target_loader)
            texts,class_labels=target_batch
            domain_labels=torch.ones(batch_size)

            if device=='cuda':
                texts=texts.cuda()
                class_labels=class_labels.cuda()
                domain_labels=domain_labels.cuda()

            _, domain_output = model(input_data=texts, alpha=alpha)
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
            cnt+=1
            if cnt%log_batch_num==0:
                print('log')

def test():
    model=UDAModel()
    with torch.inference_mode():
        for batch in test_loader:
                if device=='cuda':
                    batch = {k: v.cuda() for k, v in batch.items()}
                output = model(**batch)
                acc_num += (output.long() == batch["labels"].long()).float().sum()
        
        print(f'acc:{acc_num / len(test_loader)}\n')
    


if __name__ == '__main__':
    train()






















