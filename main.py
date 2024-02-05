import argparse
import numpy as np
from utils import get_loader_and_length
from model import UDAModel
import torch.optim as optim
import torch
import argparse

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", default='cuda', type=str, help="cpu or cuda")  # 选择运行硬件 ！cpu还没适配
parser.add_argument("--epoch_num", default=5, type=int, help="training epoch num")  # 选择训练几轮
parser.add_argument("--batch_size", default=10, type=int, help="batch size")  # batch的大小
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")  # 学习率大小
parser.add_argument("--load_epoch", default=0, type=int,
                    help="the epoch num of trained model")  # 加载之前训练过的模型，输入数字，代表选择加载已经训练了load_epoch轮的模型继续训练
parser.add_argument("--log_batch_num", default=10, type=int,
                    help="log step")  # 每log_batch_num个batch在终端输出一次训练中间的err或者其他内容
parser.add_argument("--source_path", default='data/civil_train.json', type=str, help="source dataset")  # 源域数据集
parser.add_argument("--target_path", default='data/criminal_train.json', type=str, help="target dataset")  # 目标域数据集
parser.add_argument("--test_path", default='data/criminal_test.json', type=str, help="test dataset")  # 测试集
parser.add_argument("--LLM_path", default='Base-LLMs/bert-base-chinese', type=str,
                    help="LLM path")  # 预训练大模型的位置（如果选择加载checkpoint中的模型则失效）
parser.add_argument("--class_classifier", default='MLP', type=str, help="class classifier name")  # 选择类别分类器模型，输入名称即可
parser.add_argument("--domain_classifier", default='MLP', type=str, help="domain classifier name")  # 选择域分类器模型，输入名称即可
parser.add_argument("--feature_extractor", default='BERT', type=str, help="feature extractor name")  # 选择特征提取器模型，输入名称即可

args = parser.parse_args()


def train():
    '''
    定义训练过程：
    
    
    '''
    # 获得源域和目标域加载器
    source_loader, source_len = get_loader_and_length(args.source_path, batch_size=args.batch_size)
    target_loader, target_len = get_loader_and_length(args.target_path, batch_size=args.batch_size)

    # 定义模型、优化器、损失函数
    model = UDAModel(args, args.load_epoch)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    # 转移到指定设备
    if args.device == 'cuda':
        model = model.cuda()
        loss_domain = loss_domain.cuda()
        loss_class = loss_class.cuda()

    for epoch in range(args.epoch_num):
        # 初始化统计变量
        cur_epoch = 0
        cnt = 0

        # 由于每一次训练中源域和目标域的数据要对应相等，所以不能直接遍历，而是一次次获取一个batch的数据，最后一个batch的样本数量如果不想同，还要修改数量较小的数据集到对应相等
        loader_len = min(len(source_loader), len(target_loader))

        # 获取迭代器
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        # 遍历迭代器
        while cnt < loader_len:
            # 对应论文中反向传播的lambda？
            p = float(cnt + epoch * loader_len) / args.epoch_num / loader_len
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 梯度清零
            optimizer.zero_grad()

            '''
            source 源域上的预测
            '''
            source_batch = next(source_iter)
            texts, class_labels = source_batch
            # class_labels = class_labels.clone().detach()
            domain_labels = torch.zeros(len(class_labels)).long()  # 原版UDA实现有问题，必须加len，最后一个batch大小可能小于batch_size

            if args.device == 'cuda':  # 转移到gpu上
                class_labels = class_labels.cuda()
                domain_labels = domain_labels.cuda()

            class_output, domain_output = model(input_data=texts, trade_off_param=alpha)
            err_s_label = loss_class(class_output, class_labels)  # 计算分类损失值
            err_s_domain = loss_domain(domain_output, domain_labels)  # 计算域判别损失值

            '''
            target 目标域上的预测
            '''
            target_batch = next(target_iter)
            texts, class_labels = target_batch
            domain_labels = torch.ones(len(class_labels)).long()

            if args.device == 'cuda':
                domain_labels = domain_labels.cuda()

            # 忽略输出的类别预测，所以目标域数据有没有标签不影响，为了保证代码的通用，甚至最好给数据随便加点标签
            _, domain_output = model(input_data=texts, trade_off_param=alpha)
            err_t_domain = loss_domain(domain_output, domain_labels)

            '''
            total loss and backward 计算总损失值并反向传播
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

        # 在测试集上测试分类准确率，观察模型的效果
        test(args.test_path, 'test')

        cur_epoch += 1

    # 保存模型，参数代表这个模型是训练多少轮后的模型
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

            # 忽略输出中的域类别
            class_output, _ = model(input_data=texts, trade_off_param=0)
            pred = torch.argmax(class_output, dim=-1)

            # 更新统计量
            acc_num += (pred.long() == class_labels.long()).float().sum()

    acc_num = int(acc_num)
    round_acc = round(acc_num / test_len, 4)
    print(f'{set_name} acc:{round_acc} correct/total: {acc_num}/{test_len}\n')


if __name__ == '__main__':
    train()  # 训练过程
    test(args.source_path, 'source')  # 在源域上测试分类准确率
