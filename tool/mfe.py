import torch
import torch.nn as nn


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MFE(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(MFE, self).__init__()
        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate2)
        self.layer4 = ASPPConv(in_channels, out_channels, rate3)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y1 = self.layer2(x)
        y2 = self.layer3(y1 + x)
        y3 = self.layer4(y2 + x)
        x_att =  y1 +  y2 +  y3
        return self.project(x_att + x)
