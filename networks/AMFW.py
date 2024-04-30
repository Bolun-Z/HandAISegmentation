import torch
import torch.nn as nn

class FeatExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1), bias=True,
                 norm='InstanceNorm', activation='LeakReLU'):
        """
            Extract the 2D or 3D features from the input feature map

        :param in_channels: channels of the input feature map
        :param out_channels: channels of the output feature map
        :param kSize: kernel size of the convolution (determined by 2D or 3D extractor)
        :param stride: stride size of the convolution (determined by 2D or 3D extractor)
        :param padding: padding of the convolution
        :param bias: bias of the convolution
        :param norm: the normalization method (default:InstanceNorm)
        :param activation: the activation method (default:LeakReLU)
        """
        super().__init__()
        self.norm = norm
        self.activation = activation
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kSize[0], stride=stride[0], padding=padding, bias=bias)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kSize[1], stride=stride[1], padding=padding, bias=bias)

        if self.norm == 'InstanceNorm':
            self.norm_1 = nn.InstanceNorm3d(out_channels, affine=True)
            self.norm_2 = nn.InstanceNorm3d(out_channels, affine=True)
        else:
            self.norm_1 = nn.BatchNorm3d(out_channels)
            self.norm_2 = nn.BatchNorm3d(out_channels)

        if self.activation == 'LeakReLU':
            self.activation_1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
            self.activation_2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        else:
            self.activation_1 = nn.ReLU(inplace=True)
            self.activation_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)

        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)
        return x

class Info_Inter_Module(nn.Module):
    def __init__(self, channel, M=2, k_size=3):
        """
            The local cross-channel information interaction attention mechanism
            for the fusion of multi-dimensional features.

        :param channels: the channels of the input feature map
        :param M: the number of input features
        :param k_size: the kernel size of the 1D conv, determining the scale of information interaction
        """
        super().__init__()
        self.M = M
        self.channel = channel
        self.gap = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

        self.convs = nn.ModuleList([])
        for i in range(self.M):
            self.convs.append(
                nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        batch_size, channel, _, _, _ = x1.shape
        feats = torch.cat([x1, x2], dim=1)
        feats = feats.view(batch_size, self.M, self.channel, feats.shape[2], feats.shape[3], feats.shape[4])

        feats_S = torch.sum(feats, dim=1)
        feats_G = self.gap(feats_S)

        feats_G = feats_G.squeeze(-1).squeeze(-1).transpose(-1, -2)
        attention_vectors = [conv(feats_G).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1) for conv in self.convs]

        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, channel, 1, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_o = torch.sum(feats * attention_vectors.expand_as(feats), dim=1)

        return feats_o