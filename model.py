import torch.nn as nn
from torch.autograd import Function
from feature_extracter.BERT import BERT
from class_classifier.MLP import MLP_CLASS_CLASSIFIER
from domain_classifier.MLP import MLP_DOMAIN_CLASSIFIER


class UDAModel(nn.Module):
    def __init__(self, LLM_path) -> None:  # !
        super(UDAModel, self).__init__()
        self.feature_extracter = BERT(LLM_path)
        self.class_classifier = MLP_CLASS_CLASSIFIER()
        self.domain_classifier = MLP_DOMAIN_CLASSIFIER()

    def forward(self, input, trade_off_param):
        feature = self.feature_extracter(input)
        reverse_feature = ReverseLayerF.apply(feature, trade_off_param)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
