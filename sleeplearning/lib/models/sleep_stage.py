from torch import nn
from torch.nn.init import xavier_normal
import torch.nn.functional as F


class SleepStage(nn.Module):
    def __init__(self, ts: dict):
        input_shape = ts['input_dim']
        dropout = ts['dropout']
        num_classes = ts['nclasses']
        super(SleepStage, self).__init__()
        kernel_size = 3
        self.dropout = dropout
        padding = (kernel_size // 2, kernel_size // 2)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=kernel_size,
                               padding=padding)
        self.fc1 = nn.Linear(32*(input_shape[1]//4)*(input_shape[2]//4), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.weights_init()
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = F.max_pool2d(x, (2, 2))
        x = self.conv1(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc3(x))
        return x