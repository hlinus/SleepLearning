from torch import nn
from torch.nn.init import xavier_normal
import torch.nn.functional as F


class DeepSleepNet(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(DeepSleepNet, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], num_classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x