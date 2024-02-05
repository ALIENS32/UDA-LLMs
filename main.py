import argparse
import numpy as np
from utils import get_loader_and_length
from model import UDAModel
import torch.optim as optim
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cuda', type=str, help="cpu or cuda")
parser.add_argument("--epoch_num", default=5, type=int, help="training epoch num")
parser.add_argument("--batch_size", default=10, type=int, help="batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--load_epoch", default=2, type=int, help="the epoch num of trained model")
parser.add_argument("--log_batch_num", default=10, type=int, help="log step")
parser.add_argument("--source_path", default='data/civil_train.json', type=str, help="source dataset")
parser.add_argument("--target_path", default='data/criminal_train.json', type=str, help="target dataset")
parser.add_argument("--test_path", default='data/criminal_test.json', type=str, help="test dataset")
parser.add_argument("--LLM_path", default='Base-LLMs/bert-base-chinese', type=str, help="LLM path")
parser.add_argument("--class_classifier", default='MLP', type=str, help="class classifier name")
parser.add_argument("--domain_classifier", default='MLP', type=str, help="domain classifier name")
parser.add_argument("--feature_extractor", default='BERT', type=str, help="feature extractor name")

args = parser.parse_args()


def train():
    '''
    定义训练过程：
    
    
    '''

    source_loader, source_len = get_loader_and_length(args.source_path, batch_size=args.batch_size)
    target_loader, target_len = get_loader_and_length(args.target_path, batch_size=args.batch_size)

    model = UDAModel(args, args.load_epoch)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if args.device == 'cuda':
        model = model.cuda()
        loss_domain = loss_domain.cuda()
        loss_class = loss_class.cuda()

    for epoch in range(args.epoch_num):
        cur_epoch = 0
        loader_len = min(len(source_loader), len(target_loader))

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        cnt = 0
        while cnt < loader_len:
            p = float(cnt + epoch * loader_len) / args.epoch_num / loader_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()

            '''
            source
            '''
            source_batch = next(source_iter)
            texts, class_labels = source_batch
            class_labels = torch.tensor(class_labels)
            domain_labels = torch.zeros(len(class_labels)).long()

            if args.device == 'cuda':
                class_labels = class_labels.cuda()
                domain_labels = domain_labels.cuda()

            class_output, domain_output = model(input_data=texts, trade_off_param=alpha)
            err_s_label = loss_class(class_output, class_labels)
            err_s_domain = loss_domain(domain_output, domain_labels)

            '''
            target
            '''
            target_batch = next(target_iter)
            texts, class_labels = target_batch
            domain_labels = torch.ones(len(class_labels)).long()

            if args.device == 'cuda':
                domain_labels = domain_labels.cuda()

            _, domain_output = model(input_data=texts, trade_off_param=alpha)
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
            if cnt % args.log_batch_num == 0:
                print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                      % (epoch, cnt, loader_len, err_s_label.cpu().data.numpy(), err_s_domain.cpu().data.numpy(),
                         err_t_domain.cpu().data.numpy()))

        test(args.test_path, 'test')

        cur_epoch += 1

    model.save(args.load_epoch + args.epoch_num)

    print('done')


def test(test_path, set_name):
    test_loader, test_len = get_loader_and_length(test_path, batch_size=args.batch_size)

    model = UDAModel(args, args.load_epoch)
    acc_num = 0
    if args.device == 'cuda':
        model = model.cuda()

    test_iter = iter(test_loader)
    with torch.inference_mode():
        for test_batch in test_iter:
            texts, class_labels = test_batch
            if args.device == 'cuda':
                class_labels = class_labels.cuda()

            class_output, _ = model(input_data=texts, trade_off_param=0)
            pred = torch.argmax(class_output, dim=-1)
            acc_num += (pred.long() == class_labels.long()).float().sum()

        acc_num = int(acc_num)
        round_acc = round(acc_num / test_len, 4)
        # acc_num += (output.long() == batch["labels"].long()).float().sum()

    print(f'{set_name} acc:{round_acc} correct/total: {acc_num}/{test_len}\n')


if __name__ == '__main__':
    train()
    test(args.source_path, 'source')
