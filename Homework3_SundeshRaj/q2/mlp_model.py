from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_ch, num_classes, dropout_ratio=0.2):
        super().__init__()
        self.hidden_1 = nn.Linear(in_ch, 64)
        self.dropout_1 = nn.Dropout(dropout_ratio)
        self.hidden_2 = nn.Linear(64, 64)
        self.dropout_2 = nn.Dropout(dropout_ratio)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.size(0)
        x = x.reshape(batch, -1)
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.output(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ConvNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        batch = x.size(0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class OneLayer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        batch = x.size(0)
        x = x.reshape(batch, -1)
        x = self.linear(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, 1)

    def forward(self, x):
        batch = x.size(0)
        x = x.reshape(batch, -1)
        x = self.linear(x)
        return x