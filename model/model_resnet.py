""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResnetBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResnetBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResnetBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ResnetBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResnetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResnetBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class deep_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample_conv = ResnetBlock(in_channels,out_channels,stride=2)

    def forward(self, x):
        return self.downsample_conv(x)

class Up_without_res(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResnetBlock(in_channels,out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels , kernel_size=2, stride=2)
            self.conv = ResnetBlock(in_channels,out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)
    
class deep_Up_without_res(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResnetBlock(in_channels,out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels , kernel_size=2, stride=2)
            self.conv = ResnetBlock(in_channels,out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    """ Full assembly of the parts to form the complete network """

class UNet_ds16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_ds16, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (deep_Down(64, 128))
        self.down2 = (deep_Down(128, 256))
        self.down3 = (deep_Down(256, 512))
        self.down4 = (deep_Down(512, 768))
        self.encoder=nn.Sequential(
            self.down1,self.down2,self.down3,self.down4
        )
        self.up1 = (deep_Up_without_res(768, 512, bilinear))
        self.up2 = (deep_Up_without_res(512, 256 , bilinear))
        self.up3 = (deep_Up_without_res(256, 128 , bilinear))
        self.up4 = (deep_Up_without_res(128, 64, bilinear))
        self.decoder=nn.Sequential(
            self.inc,self.up1,self.up2,self.up3,self.up4
        )
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # x=self.encoder(x)
        # x=self.decoder(x)
        logits = self.outc(x)
        return logits

class UNet_ds32(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_ds32, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (deep_Down(64, 64))
        self.down2 = (deep_Down(64, 128))
        self.down3 = (deep_Down(128, 256))
        self.down4 = (deep_Down(256, 512))
        self.down5 = (deep_Down(512, 768))
        self.encoder=nn.Sequential(
            self.down1,self.down2,self.down3,self.down4,self.down5
        )
        self.up1 = (deep_Up_without_res(768, 512, bilinear))
        self.up2 = (deep_Up_without_res(512, 256 , bilinear))
        self.up3 = (deep_Up_without_res(256, 128 , bilinear))
        self.up4 = (deep_Up_without_res(128, 64, bilinear))
        self.up5 = (deep_Up_without_res(64, 64, bilinear))
        self.decoder=nn.Sequential(
            self.up1,self.up2,self.up3,self.up4,self.up5
        )
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5)
        # x = self.up2(x)
        # x = self.up3(x)
        # x = self.up4(x)
        x=self.encoder(x1)
        x=self.decoder(x)
        logits = self.outc(x)
        return logits
    
class UNet_ds64(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_ds32, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (deep_Down(64, 64))
        self.down2 = (deep_Down(64, 128))
        self.down3 = (deep_Down(128, 256))
        self.down4 = (deep_Down(256, 512))
        self.down5 = (deep_Down(512, 768))
        self.down6 = (deep_Down(768, 768))
        self.encoder=nn.Sequential(
            self.down1,self.down2,self.down3,self.down4,self.down5,self.down6
        )
        self.up1 = (deep_Up_without_res(768, 768, bilinear))
        self.up2 = (deep_Up_without_res(768, 512, bilinear))
        self.up3 = (deep_Up_without_res(512, 256 , bilinear))
        self.up4 = (deep_Up_without_res(256, 128 , bilinear))
        self.up5 = (deep_Up_without_res(128, 64, bilinear))
        self.up6 = (deep_Up_without_res(64, 64, bilinear))
        self.decoder=nn.Sequential(
            self.up1,self.up2,self.up3,self.up4,self.up5,self.up6
        )
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5)
        # x = self.up2(x)
        # x = self.up3(x)
        # x = self.up4(x)
        x=self.encoder(x1)
        x=self.decoder(x)
        logits = self.outc(x)
        return logits
