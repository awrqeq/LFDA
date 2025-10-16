import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """ (Convolutional) Channel Attention Block """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class MultiScaleBlock(nn.Module):
    """ Multi-Scale Convolutional Block """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        # 1x1 conv to fuse the concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch3 = self.branch3x3(x)
        branch5 = self.branch5x5(x)

        concatenated = torch.cat([branch1, branch3, branch5], 1)
        return self.fuse(concatenated)


class MultiScaleAttentionGenerator(nn.Module):
    """
    升级后的触发器生成器 (MSAG)，结合了多尺度卷积和通道注意力。
    输入：DWT后的高频子带 (e.g., HH sub-band)
    输出：对应形状的频域扰动
    """

    def __init__(self, in_channels=3, base_channels=64):
        super(MultiScaleAttentionGenerator, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.ca1 = ChannelAttention(base_channels)
        self.msb1 = MultiScaleBlock(base_channels, base_channels)

        self.ca2 = ChannelAttention(base_channels)
        self.msb2 = MultiScaleBlock(base_channels, base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, hh_subband):
        x = self.initial_conv(hh_subband)

        x = self.ca1(x) * x
        x = self.msb1(x)

        x = self.ca2(x) * x
        x = self.msb2(x)

        output = self.final_conv(x)

        # 使用Tanh将输出范围限制在[-1, 1]，方便后续进行强度缩放
        return self.tanh(output)