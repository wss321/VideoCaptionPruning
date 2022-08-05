import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class FilterImportanceEstimater(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, kernel_size=3, hidden_dim=None):
        super(FilterImportanceEstimater, self).__init__()
        dim = in_channel
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            for k in kernel_size:
                dim *= k
        elif isinstance(kernel_size, int):
            dim *= kernel_size ** 2
        self.dim = dim
        if hidden_dim is None:
            hidden_dim = in_channel
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.hidden_dim),
            nn.GELU(),
        )
        self.att = nn.Sequential(
            nn.Linear(out_channel, out_channel),
            nn.GELU(),
        )
        self.cnt = 0
        # n_head = 1
        # self.att = MultiHeadAttention(n_head, 128, 128//n_head, 128//n_head)

    def forward(self, conv_weight):
        # num, channel, h, w
        x = conv_weight.view(-1, self.dim)
        x = self.fc(x)
        x = torch.matmul(x / (x.size(1)**0.5), x.transpose(0, 1))
        # x = self.att(x.transpose(0, 1))
        # m = torch.min(x, dim=0)[0]
        # M = torch.max(x, dim=0)[0]
        x = torch.sum(torch.abs(x), dim=1) / x.size(1)
        # x = torch.max(x, dim=0)[0]  # max
        # x = torch.sum(x, dim=0)  # sum

        x = (x-x.min())/(x.max()-x.min() + 1e-8)  # normalize
        # x = x - x.mean()
        # m = torch.min(x, dim=0)[0]
        # M = torch.max(x, dim=0)[0]
        # print()
        # x = (x - m) / (M - m)
        # print(x.shape)
        # x = torch.softmax(x, 0)
        # x = torch.sigmoid(x)
        if self.cnt % 100 == 0:
            print(x, torch.var(x))
        x = x.view(1, x.size(0), 1, 1)
        self.cnt += 1
        return x
    # def forward(self, conv_weight):
    #     # num, channel, h, w
    #     x = conv_weight.view(-1, self.dim)
    #     x = self.fc(x)
    #     x = x.unsqueeze(0)
    #     x = self.att(x, x, x).squeeze(0)
    #     x = torch.sum(x, dim=1)
    #     x = (x-x.min())/(x.max()-x.min())  # normalize
    #     x = x.view(1, x.size(0), 1, 1)
    #     return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.fle1 = FilterImportanceEstimater(in_planes, planes, kernel_size=3, hidden_dim=None)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.fle2 = FilterImportanceEstimater(planes, planes, kernel_size=3, hidden_dim=None)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        c = self.conv1(x)
        c = c * self.fle1(self.conv1.weight) + c
        out = F.relu(self.bn1(c))
        c = self.conv2(out)
        c = c * self.fle2(self.conv2.weight) + c
        out = self.bn2(c)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.fle1 = FilterImportanceEstimater(in_planes, planes, kernel_size=1, hidden_dim=in_planes // 4)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.fle2 = FilterImportanceEstimater(planes, planes, kernel_size=3, hidden_dim=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.fle3 = FilterImportanceEstimater(planes, self.expansion * planes, kernel_size=1, hidden_dim=planes // 4)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        c = self.conv1(x)
        c = c * self.fle1(self.conv1.weight) + c
        out = F.relu(self.bn1(c))

        c = self.conv2(out)
        c = c * self.fle2(self.conv2.weight) + c
        out = F.relu(self.bn2(c))

        c = self.conv3(out)
        c = c * self.fle3(self.conv3.weight) + c
        out = self.bn3(c)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out, feature


def resnet18(num_classes=10, **kws):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10, **kws):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=10, **kws):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=10, **kws):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=10, **kws):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
