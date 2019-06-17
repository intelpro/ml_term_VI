"""toynet.py"""
import torch.nn as nn
import torch.nn.functional as F
class ToyNet_MNIST(nn.Module):
    def __init__(self, x_dim=784, y_dim=10):
        super(ToyNet_MNIST, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim, 300),
            nn.ReLU(True),
            nn.Linear(300, 150),
            nn.ReLU(True),
            nn.Linear(150, self.y_dim)
            )

    def forward(self, X):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        out = self.mlp(X)

        return out

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

# I added it
class ToyNet_CIFAR10(nn.Module):
    def __init__(self, h_dim=32, w_dim=32, ch_dim =3, y_dim=10):
        super(ToyNet_CIFAR10, self).__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.ch_dim = ch_dim
        self.y_dim = y_dim

        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(-1,16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())


# Can change initialization kaiming to xavier
def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()
