import torch.nn as nn
import torch


class MLP_CLASS_CLASSIFIER(nn.Module):
    def __init__(self):
        super(MLP_CLASS_CLASSIFIER, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(768, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

    def forward(self, input_feature):
        class_output = self.class_classifier(input_feature)
        return class_output
    
    def save_model(self,epoch_num):
        torch.save(self.class_classifier.state_dict(), f'checkpoint/class_classifier/MLP/{epoch_num}.pth')

    def load_model(self,epoch_num):
        state_dict = torch.load(f'checkpoint/class_classifier/MLP/{epoch_num}.pth')
        self.class_classifier.load_state_dict(state_dict)
