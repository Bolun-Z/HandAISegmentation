import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from networks.AMFW import FeatExtractor, Info_Inter_Module

class Layer_Down(nn.Module):
    def __init__(self, in_channels, out_channels, min_z=8, downsample=True):
        """
        basic module at downsampling stage

        :param in_channels: channels of the input feature map
        :param out_channels: channels of the output feature map
        :param min_z: if the size of z-axis < min_z, max_pooling won't be applied along z-axis
        :param downsample: perform down sample or not
        """
        super().__init__()
        self.min_z = min_z
        self.downsample = downsample
        self.Feat_extractor_2D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                         kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1))

        self.Feat_extractor_3D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                         kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1))

        self.IIM = Info_Inter_Module(channel=out_channels)

    def forward(self, x):
        if self.downsample:
            if x.shape[2] >= self.min_z:
                x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            else:
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        x = self.IIM(self.Feat_extractor_2D(x), self.Feat_extractor_3D(x))

        return x


class Layer_Up(nn.Module):
    def __init__(self, in_channels, out_channels, SKIP=True):
        """
        basic module at upsampling stage

        :param in_channels: channels of the input feature map
        :param out_channels: channels of the output feature map
        :param SKIP: there are skip connection or not
        """
        super().__init__()
        self.SKIP = SKIP

        self.Feat_extractor_2D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                         kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1))

        self.Feat_extractor_3D = FeatExtractor(in_channels=in_channels, out_channels=out_channels,
                         kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1))

        self.IIM = Info_Inter_Module(channel=out_channels)

    def forward(self, x):
        if self.SKIP:
            x, xskip = x
            tarSize = xskip.shape[2:]
            up = F.interpolate(x, size=tarSize, mode='trilinear', align_corners=False)
            cat = torch.cat([xskip, up], dim=1)
            x = self.IIM(self.Feat_extractor_2D(cat), self.Feat_extractor_3D(cat))
        else:
            x = self.IIM(self.Feat_extractor_2D(x), self.Feat_extractor_3D(x))
        return x

class AMWNet(nn.Module):
    def __init__(self, in_channel=1, n_classes=2):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(AMWNet, self).__init__()

        self.ec_layer1 = Layer_Down(self.in_channel, 64, downsample=False)
        self.ec_layer2 = Layer_Down(64, 128)
        self.ec_layer3 = Layer_Down(128, 256)
        self.ec_layer4 = Layer_Down(256, 512)

        self.dc_layer4 = Layer_Up(256 + 512, 256)
        self.dc_layer3 = Layer_Up(128 + 256, 128)
        self.dc_layer2 = Layer_Up(64 + 128, 64)
        self.dc_layer1 = Layer_Up(64, 32, SKIP=False)
        self.dc_layer0 = nn.Conv3d(32, n_classes, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)

        self.pool0 = nn.MaxPool3d(2)

    def encoder(self, x):
        with autocast():
            # x = x.view(x.shape[0], x.shape[1], 8, 16, 16)  # sample 2048
            x = x.view(x.shape[0], x.shape[1], 8, 8, 16) # sample 1024
            # x = x.view(x.shape[0], x.shape[1], 8, 8, 8) # sample 512
            feat_1 = self.pool0(x)
            feat_2 = self.ec_layer2(feat_1)
            feat_3 = self.ec_layer3(feat_2)

        return feat_3

    def forward(self, x):
        with autocast():
            feat_i = self.ec_layer1(x)

            feat_1 = self.pool0(feat_i)
            feat_2 = self.ec_layer2(feat_1)
            feat_3 = self.ec_layer3(feat_2)
            feat_4 = self.ec_layer4(feat_3)

            dfeat_4 = self.dc_layer4([feat_4, feat_3])
            dfeat_3 = self.dc_layer3([dfeat_4, feat_2])
            dfeat_2 = self.dc_layer2([dfeat_3, feat_i])
            dfeat_1 = self.dc_layer1(dfeat_2)
            output = self.dc_layer0(dfeat_1)

            return feat_i, output