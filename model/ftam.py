import torch
import torch.nn as nn


class FeaturemapAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(FeaturemapAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TemporalAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class FTAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(FTAM, self).__init__()
        self.fa = FeaturemapAttention(in_planes, ratio)
        self.ta = TemporalAttention(kernel_size)

    def forward(self, x):
        out = x * self.fa(x)
        result = out * self.ta(out)
        return result
