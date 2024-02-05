import torch
import torch.nn as nn
from torch.autograd import Function
import importlib
import os


class UDAModel(nn.Module):
    def __init__(self, args, epoch_num=None) -> None:
        super(UDAModel, self).__init__()

        # 父文件夹路径，主要在load和save函数中使用
        self.parent_file_path = 'checkpoint/' + args.feature_extractor + '_' + args.class_classifier + '_' + args.domain_classifier + '/'

        '''
        定义分类器、域判别器、特征提取器
        '''
        # 这6行为了根据超参数动态的导入模块中的类，可以动态的指定模块，方便模型组合模块，但是需要各个模块在定义的时候满足一定规范，按照我的类定义写就行
        class_classifier_module = importlib.import_module('class_classifier.' + args.class_classifier)
        domain_classifier_module = importlib.import_module('domain_classifier.' + args.domain_classifier)
        feature_extractor_module = importlib.import_module('feature_extractor.' + args.feature_extractor)

        class_classifier = getattr(class_classifier_module, args.class_classifier + '_CLASS_CLASSIFIER')
        domain_classifier = getattr(domain_classifier_module, args.domain_classifier + '_DOMAIN_CLASSIFIER')
        feature_extractor = getattr(feature_extractor_module, args.feature_extractor + '_FEATURE_EXTRACTOR')

        self.class_classifier = class_classifier(args)
        self.domain_classifier = domain_classifier(args)
        self.feature_extractor = feature_extractor(args)

        # 如果要加载之前训练过的模型，利用load接口执行load函数，会覆盖前面定义的模型或者加载参数
        if epoch_num != 0:
            self.load(epoch_num=epoch_num)

    def forward(self, input_data, trade_off_param):
        '''
        前向传播
        '''
        feature = self.feature_extractor(input_data)  # 输出特征
        reverse_feature = ReverseLayerF.apply(feature, trade_off_param)  # 给来自源域的梯度乘上权重
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def save(self, epoch_num):
        '''
        保存模型
        '''
        os.makedirs(self.parent_file_path, exist_ok=True)  # 只有在第一次保存这个组合的UDA模型是会创建文件夹，后面再存，这个文件夹就已经存在了，不用再创建了
        model_file_path = self.parent_file_path + str(epoch_num)
        os.makedirs(self.parent_file_path + str(epoch_num), exist_ok=True)  # 根据epoch_num建立当前模型存放的父级文件夹，例如BERT_MLP_MLP_5，代表训练了5抡的BERT_MLP_MLP组合的模型
        # 调用各个模块模型的接口存储模型
        self.feature_extractor.save_model(model_file_path)
        self.class_classifier.save_model(model_file_path)
        self.domain_classifier.save_model(model_file_path)

    def load(self, epoch_num):
        # 根据epoch_num确认父级文件夹
        model_file_path = self.parent_file_path + str(epoch_num)
        # 调用各自模型的接口加载模型
        self.feature_extractor.load_model(model_file_path)
        self.class_classifier.load_model(model_file_path)
        self.domain_classifier.load_model(model_file_path)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
