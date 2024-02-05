import torch.nn as nn

class MLP_DOMAIN_CLASSIFIER(nn.Module):
    def __init__(self):
        super(MLP_DOMAIN_CLASSIFIER,self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(768, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self,input_feature):
        domain_output = self.domain_classifier(input_feature)
        return domain_output

