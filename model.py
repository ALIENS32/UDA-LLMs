import torch
import torch.nn as nn
from torch.autograd import Function
import importlib
import os


class UDAModel(nn.Module):
    def __init__(self, args, epoch_num=None) -> None:
        super(UDAModel, self).__init__()

        self.parent_file_path = 'checkpoint/' + args.feature_extractor + '_' + args.class_classifier + '_' + args.domain_classifier + '/'

        class_classifier_module = importlib.import_module('class_classifier.' + args.class_classifier)
        domain_classifier_module = importlib.import_module('domain_classifier.' + args.domain_classifier)
        feature_extractor_module = importlib.import_module('feature_extractor.' + args.feature_extractor)

        class_classifier = getattr(class_classifier_module, args.class_classifier + '_CLASS_CLASSIFIER')
        domain_classifier = getattr(domain_classifier_module, args.domain_classifier + '_DOMAIN_CLASSIFIER')
        feature_extractor = getattr(feature_extractor_module, args.feature_extractor + '_FEATURE_EXTRACTOR')

        self.class_classifier = class_classifier(args)
        self.domain_classifier = domain_classifier(args)
        self.feature_extractor = feature_extractor(args)

        if epoch_num != 0:
            self.load(epoch_num=epoch_num)

    def forward(self, input_data, trade_off_param):
        feature = self.feature_extractor(input_data)
        reverse_feature = ReverseLayerF.apply(feature, trade_off_param)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

    def save(self, epoch_num):
        os.makedirs(self.parent_file_path, exist_ok=True)
        model_file_path = self.parent_file_path + str(epoch_num)
        os.makedirs(self.parent_file_path + str(epoch_num), exist_ok=True)
        self.feature_extractor.save_model(model_file_path)
        self.class_classifier.save_model(model_file_path)
        self.domain_classifier.save_model(model_file_path)

    def load(self, epoch_num):
        model_file_path = self.parent_file_path + str(epoch_num)
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
