import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal


class Net(nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3)
        self.fc1 = nn.Linear(14080, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = F.max_pool2d(x, (2, 3))
        x = self.conv1(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc2(x))
        return x

