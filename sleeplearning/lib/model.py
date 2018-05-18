import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal


class Net(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(Net, self).__init__()
        kernel_size = 3
        padding = (kernel_size // 2, kernel_size // 2)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=kernel_size,
                               padding=padding)
        self.fc1 = nn.Linear(32*(input_shape[1]//4)*(input_shape[2]//4), 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

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
        #x = F.relu(self.fc2(x))
        #x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.fc3(x))
        return x


class Overfit(nn.Module):
    def __init__(self, num_classes: int, input_shape: tuple):
        super(Overfit, self).__init__()
        kernel_size = 3
        padding = (kernel_size//2, kernel_size//2)
        self.conv1 = nn.Conv2d(input_shape[0], 48, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(48, 32, kernel_size=3, padding=padding)
        self.fc1 = nn.Linear(32*input_shape[1]//4*input_shape[2]//4, 8000)
        self.fc2 = nn.Linear(8000, 5000)
        self.fc3 = nn.Linear(5000, 3000)
        self.fc4 = nn.Linear(3000, 1000)
        self.fc5 = nn.Linear(1000, num_classes)

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            xavier_normal(m.weight.data)
            xavier_normal(m.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv2(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # Flatten layer

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


