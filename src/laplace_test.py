import torch
import torch.distributions as dists
import numpy as np
from netcal.metrics import ECE

from laplace import Laplace


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import urllib.request
import os.path
import matplotlib.pyplot as plt

# Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils


train_batch_size = 128
test_batch_size = 100


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def CIFAR10(data_dir):

    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    training_data = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=128,
                                         shuffle=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                         shuffle=False, num_workers=0)

    return train_loader, test_loader

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = self.conv2(out)

        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []

        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes, num_channel=3, dropRate=0.3, feature_extractor=False):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(num_channel, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)

        self.nChannels = nChannels[3]
        self.feature_extractor = feature_extractor

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)

        if self.feature_extractor:
            return out

        return self.fc(out)


    def features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out


np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

train_loader, test_loader = CIFAR10(data_dir="./data/")

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()

# The model is a standard WideResNet 16-4
# Taken as is from https://github.com/hendrycks/outlier-exposure
model = WideResNet(16, 4, num_classes=10).eval()

model.load_state_dict(torch.load('./temp/CIFAR10_plain.pt', map_location=torch.device('cpu')))

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))

    return torch.cat(py).cpu().numpy()


probs_map = predict(test_loader, model, laplace=False)
acc_map = (probs_map.argmax(-1) == targets).mean()
ece_map = ECE(bins=15).measure(probs_map, targets)
nll_map = -dists.Categorical(torch.tensor(probs_map)).log_prob(torch.tensor(targets)).mean()

print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='diag')
la.fit(train_loader)
la.optimize_prior_precision(method='marglik')

probs_laplace = predict(test_loader, la, laplace=True)
acc_laplace = (probs_laplace.argmax(-1) == targets).mean()
ece_laplace = ECE(bins=15).measure(probs_laplace, targets)
nll_laplace = -dists.Categorical(torch.tensor(probs_laplace)).log_prob(torch.tensor(targets)).mean()

print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')