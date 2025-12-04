# model_unet.py
# PyTorch implementation of a 2D U-Net for TF-mask estimation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscale with MaxPool then DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscale then DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # use simple ConvTranspose2d upsampling
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1: from decoder (low-res), x2: skip connection (high-res)
        x1 = self.up(x1)

        # pad if necessary (in case of odd sizes)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for real-valued TF-mask prediction.
    Input:  (B, 1, F, T) magnitude spectrogram
    Output: (B, 1, F, T) mask in [0, 1] (Sigmoid)
    """

    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.bottom = DoubleConv(base_ch * 8, base_ch * 16)

        self.up3 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up0 = Up(base_ch * 2, base_ch)

        self.outc = nn.Conv2d(base_ch, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()  # mask in [0,1]

    def forward(self, x):
        # encoder
        x0 = self.inc(x)       # (B, 64, F, T)
        x1 = self.down1(x0)    # (B, 128, F/2, T/2)
        x2 = self.down2(x1)    # (B, 256, F/4, T/4)
        x3 = self.down3(x2)    # (B, 512, F/8, T/8)

        # bottom
        xb = self.bottom(x3)

        # decoder with skip-connections
        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        x = self.outc(x)
        x = self.act(x)
        return x
