# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from .layers import ConvBlock3x3, Conv3x3, upsample, ConvBlock1x1

from packnet_sfm.networks.depth.FPTResNet.modeling.FPT import FPT
from packnet_sfm.networks.depth.FPTResNet.modeling.FPT_GTOnly import FPT_GTOnly
from packnet_sfm.networks.depth.FPTResNet.modeling.FPT_RTOnly import FPT_RTOnly
from packnet_sfm.networks.depth.FPTResNet.modeling.FPT_STOnly import FPT_STOnly
from packnet_sfm.networks.depth.FPTResNet.modeling.FPT_GT_RT import FPT_GT_RT
from packnet_sfm.networks.depth.FPTResNet.modeling.DenseFPT import DenseFPT

from packnet_sfm.networks.depth.FPTResNet.models.axialnet import AxialAttention

from packnet_sfm.networks.depth.FPTResNet.cvpods.layers import TreeFilterV2

from packnet_sfm.networks.depth.FPTResNet.standalone.attention import AttentionConv


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, mode=None):
        super(DepthDecoder, self).__init__()

        mode = mode.split(' ')
        self.mode = mode[0]
        self.network = mode[1]

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # self.num_ch_enc[1:] = 256
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.DenseFPT = OrderedDict()
        self.block = OrderedDict()

        # 中间层特征的语义增强
        # self.FPT_2 = DenseFPT(feature_dim=128, num_feature_maps=3, channels=[2048, 1024, 512])

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock3x3(num_ch_in, num_ch_out)

            if 5 <= i <= 4:
                # channels = [num_ch_enc[-1], *self.num_ch_dec[i+1:][::-1], num_ch_enc[i - 1]]
                channels = [*num_ch_enc[i+1:], num_ch_in, num_ch_enc[i - 1]]
                self.DenseFPT[i] = DenseFPT(feature_dim=num_ch_out, num_feature_maps=6-i, channels=channels[:6-i])

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if 5 <= i <= 4:
                pass
            else:
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
            if i+1 in self.scales:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock3x3(num_ch_in, num_ch_out)

            if 0 <= i <= 4:
                # # 语义特征融合
                # self.block[('hight', i)] = AxialAttention(num_ch_out, num_ch_out, num_ch_out, groups=2, kernel_size=12*2**(4-i))
                # self.block[('width', i)] = AxialAttention(num_ch_out, num_ch_out, num_ch_out, groups=2, kernel_size=40*2**(4-i), width=True)
                # self.block[('nonlinear', i)] = nn.ELU(inplace=True)

                # self.block[('semantic_weight', i)] = ConvBlock1x1(num_ch_out, 2)

                # # 光流特征融合
                # self.block[('hight', i)] = AxialAttention(2, num_ch_out, num_ch_out, groups=2, kernel_size=12*2**(4-i))
                # self.block[('width', i)] = AxialAttention(2, num_ch_out, num_ch_out, groups=2, kernel_size=40*2**(4-i), width=True)
                # self.block[('nonlinear', i)] = nn.ELU(inplace=True)

                # # edge特征
                # self.block[('edge', i)] = TreeFilterV2(21, num_ch_out, guide_embed_channels=21, in_embed_channels=num_ch_out)
                # 语义特征
                self.block[('semantic', i)] = TreeFilterV2(num_ch_out, num_ch_out, guide_embed_channels=num_ch_out//2, in_embed_channels=num_ch_out//2)

                # # 取临近区域的特征
                # self.block[('standalone', i)] = AttentionConv(in_channels=num_ch_out, out_channels=num_ch_out, kernel_size=7, padding=3, groups=8)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.DenseFPT2cuda = nn.ModuleList(list(self.DenseFPT.values()))
        self.block2cuda = nn.ModuleList(list(self.block.values()))
        self.sigmoid = nn.Sigmoid()

        # self.FPT = FPT(256)
        # self.RT = FPT_RTOnly(256)
        # self.GT = FPT_GTOnly(256)
        # self.ST = FPT_STOnly(256)
        # self.GT_RT = FPT_GT_RT(256)

        # self.hight_block_2_4 = AxialAttention(512, 2048, 256, groups=8, kernel_size=24)
        # self.hight_block_2_3 = AxialAttention(512, 1024, 256, groups=8, kernel_size=24)
        # self.hight_block_2_2 = AxialAttention(512, 512, 256, groups=8, kernel_size=24)
        # self.width_block_2_4 = AxialAttention(512, 2048, 256, groups=8, kernel_size=80, width=True)
        # self.width_block_2_3 = AxialAttention(512, 1024, 256, groups=8, kernel_size=80, width=True)
        # self.width_block_2_2 = AxialAttention(512, 512, 256, groups=8, kernel_size=80, width=True)
        # self.refine1x1_2_4 = ConvBlock1x1(512, 128)
        # self.refine1x1_2_3 = ConvBlock1x1(512, 128)
        # self.refine1x1_2_2 = ConvBlock1x1(512, 128)
        # self.refine1x1_2 = ConvBlock1x1(512, 128)
        # self.enhance_2 = ConvBlock3x3(128*4, 128)

        # self.hight_block_2_4 = AxialAttention(512, 2048, 256, groups=8, kernel_size=24)
        # self.hight_block_2_3 = AxialAttention(512, 1024, 256, groups=8, kernel_size=24)
        # self.hight_block_2_2 = AxialAttention(512, 512, 256, groups=8, kernel_size=24)
        # self.width_block_2_4 = AxialAttention(512, 256, 256, groups=8, kernel_size=80, width=True)
        # self.width_block_2_3 = AxialAttention(512, 256, 256, groups=8, kernel_size=80, width=True)
        # self.width_block_2_2 = AxialAttention(512, 256, 256, groups=8, kernel_size=80, width=True)
        # self.refine1x1_2_4 = ConvBlock1x1(256, 128)
        # self.refine1x1_2_3 = ConvBlock1x1(256, 128)
        # self.refine1x1_2_2 = ConvBlock1x1(256, 128)
        # self.refine1x1_2 = ConvBlock1x1(512, 128)
        # self.enhance_2 = ConvBlock3x3(128*4, 128)

    def semantic_enhance(self, features):
        out = features[:3]

        # # FPT基本为GT
        # out[2] = self.FPT_2(features[2:][::-1])

        # 十字形Transformer
        features[4] = F.interpolate(features[4], scale_factor=4, mode='bilinear')
        features[3] = F.interpolate(features[3], scale_factor=2, mode='bilinear')

        # out_2_4_hight = self.hight_block_2_4(out[2], features[4])
        # out_2_4_width = self.width_block_2_4(out[2], features[4])
        # out_2_4 = self.refine1x1_2_4(torch.cat((out_2_4_hight, out_2_4_width), 1))

        # out_2_3_hight = self.hight_block_2_3(out[2], features[3])
        # out_2_3_width = self.width_block_2_3(out[2], features[3])
        # out_2_3 = self.refine1x1_2_3(torch.cat((out_2_3_hight, out_2_3_width), 1))

        # out_2_2_hight = self.hight_block_2_2(out[2], features[2])
        # out_2_2_width = self.width_block_2_2(out[2], features[2])
        # out_2_2 = self.refine1x1_2_2(torch.cat((out_2_2_hight, out_2_2_width), 1))

        # out[2] = torch.cat([out_2_4, out_2_3, out_2_2, self.refine1x1_2(out[2])], 1)
        # out[2] = self.enhance_2(out[2])


        out_2_4_hight = self.hight_block_2_4(out[2], features[4])
        out_2_4_width = self.width_block_2_4(out[2], out_2_4_hight)
        out_2_4 = self.refine1x1_2_4(out_2_4_width)

        out_2_3_hight = self.hight_block_2_3(out[2], features[3])
        out_2_3_width = self.width_block_2_3(out[2], out_2_3_hight)
        out_2_3 = self.refine1x1_2_3(out_2_3_width)

        out_2_2_hight = self.hight_block_2_2(out[2], features[2])
        out_2_2_width = self.width_block_2_2(out[2], out_2_2_hight)
        out_2_2 = self.refine1x1_2_2(out_2_2_width)

        out[2] = torch.cat([out_2_4, out_2_3, out_2_2, self.refine1x1_2(out[2])], 1)
        out[2] = self.enhance_2(out[2])

        return out

    def forward(self, input_features, other_features=None):
        # # 使用全部的FPT操作
        # input_features[1:] = self.FPT(*input_features[1:])
        # # 仅使用FPT中的RT操作
        # input_features[1:] = self.RT(*input_features[1:])
        # # 仅使用FPT中的GT操作
        # input_features[1:] = self.GT(*input_features[1:])
        # # 仅使用FPT中的ST操作
        # input_features[1:] = self.ST(*input_features[1:])
        # 使用FPT中的GT+RT操作
        # input_features[1:] = self.GT_RT(*input_features[1:])
        # # 增强中间层，得到3个特征
        # input_features = self.semantic_enhance(input_features)

        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            if 5 <= i <= 4:
                x = [self.DenseFPT[i]([*input_features[i+1:], x, input_features[i - 1]])]
            else:
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
            if i+1 in self.scales:
                x += [upsample(self.outputs[("disp", i+1)])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if 0 <= i <= 4:
                # transformer特征融合
                if self.mode == 'axial':
                    if self.network == 'SGDepth':
                        # SGDepth
                        identity = x
                        
                        # x = self.block[('hight', i)](x, other_features[i])
                        # x = self.block[('width', i)](x, other_features[i])
                        # x = self.block[('nonlinear', i)](x)

                        x0 = self.block[('hight', i)](x, other_features[i])
                        x1 = self.block[('width', i)](x, other_features[i])
                        semantic_weight = F.softmax(self.block[('semantic_weight', i)](other_features[i]), 1)
                        x = semantic_weight[:, 0, :, :].unsqueeze(1)*x0 + semantic_weight[:, 1, :, :].unsqueeze(1)*x1
                        x = self.block[('nonlinear', i)](x)

                        x = x + identity

                    elif self.network == 'optical':
                        # optical
                        identity = x
                        x = self.block[('hight', i)](x, other_features[i])
                        x = self.block[('width', i)](x, other_features[i])
                        x = self.block[('nonlinear', i)](x)
                        x = x + identity

                elif self.mode == 'standalone':
                    if self.network == 'SGDepth':
                        # SGDepth
                        identity = x
                        x = self.block[('standalone', i)](x, other_features[i])
                        x = x + identity

                elif self.mode == 'tree':
                    if self.network == 'BDCN':
                        x = self.block[('edge', i)](x, other_features[i])
                    elif self.network == 'SGDepth':
                        x = self.block[('semantic', i)](x, other_features[i])

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
