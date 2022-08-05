import torch
import torch.nn as nn
from models.fle_resnet import resnet50

a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
b = resnet50()
for k, v in b.named_parameters():
    print(k)
# print(a.weight.shape)
# print(a.weight[0, 0])
# print(a.weight[0, 1])
