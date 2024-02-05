import torch.nn as nn
import torch


class MLP_DOMAIN_CLASSIFIER(nn.Module):
    def __init__(self, args):
        super(MLP_DOMAIN_CLASSIFIER, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(768, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_feature):
        domain_output = self.domain_classifier(input_feature)
        return domain_output

    def save_model(self, model_file_path):
        torch.save(self.domain_classifier.state_dict(), model_file_path + '/MLP_DOMAIN_CLASSIFIER.pth')

    def load_model(self, model_file_path):
        state_dict = torch.load(model_file_path + '/MLP_DOMAIN_CLASSIFIER.pth')
        self.domain_classifier.load_state_dict(state_dict)
