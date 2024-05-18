import torch
from torch import nn
from torch.nn import functional as F

class RenderTrans(nn.Module):
    def __init__(self, channels_high, channels_low, level, upsample=True):
        super(RenderTrans, self).__init__()
        self.level = level
        self.upsample = upsample

        self.conv3x3 = nn.Conv2d(channels_high, channels_high, kernel_size=3, padding=1, bias=False)
        # self.bn_low = nn.BatchNorm2d(channels_high)
        self.GN_low = nn.GroupNorm(16, channels_high)

        self.conv1x1 = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_high)
        # self.GN_high = nn.GroupNorm(16, channels_high)

        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_low, channels_high, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_high)
        else:
            self.conv_reduction = nn.Conv2d(channels_low, channels_high, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_high)

        str_conv3x3_list = []
        for i in range(level):
            str_conv3x3_list.append(nn.Conv2d(channels_low, channels_high, kernel_size=3, stride=2, padding=1, bias=False))
        self.str_conv3x3 = nn.ModuleList(str_conv3x3_list)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Conv2d(channels_high*2, channels_high, kernel_size=1, padding=0, bias=False)

    def forward(self, x_high, x_low):
        b, c, h, w = x_low.shape
        x_low_gp = nn.AvgPool2d(x_low.shape[2:])(x_low).view(len(x_low), c, 1, 1)
        x_low_gp = self.conv1x1(x_low_gp)
        # x_low_gp = self.bn_low(x_low_gp)
        x_low_gp = self.GN_low(x_low_gp)
        x_low_gp = self.relu(x_low_gp)

        x_high_mask = self.conv3x3(x_high)
        x_high_mask = self.bn_high(x_high_mask)
        # x_high_mask = self.GN_high(x_high_mask)

        x_att = x_high_mask * x_low_gp

        x_low_v = x_low
        for i in range(self.level):
            x_low_v = self.str_conv3x3[i](x_low_v)

        if self.upsample:
            out = self.relu(
                self.bn_upsample(x_low_v) + x_att)
                # self.conv_cat(torch.cat([self.bn_upsample(self.str_conv3x3(x_low)), x_att], dim=1))
        else:
            out = self.relu(
                self.bn_reduction(x_low_v) + x_att)
                # self.conv_cat(torch.cat([self.bn_reduction(self.str_conv3x3(x_low)), x_att], dim=1))
        return out
