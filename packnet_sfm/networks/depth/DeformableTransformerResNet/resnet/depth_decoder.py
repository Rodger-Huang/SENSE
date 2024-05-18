# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .layers import ConvBlock3x3, Conv3x3, upsample, ConvBlock1x1

from packnet_sfm.networks.depth.DeformableTransformerResNet.transform import transform

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.H = np.array([96, 48, 24, 12, 6])
        self.W = np.array([320, 160, 80, 40, 20])
        self.p = np.array([16, 8, 4, 2, 1])
        # decoder
        self.convs = OrderedDict()
        self.deformableTransforms = OrderedDict()

        # 中间层特征的语义增强
        self.deTrans_2_4 = transform(num_channels=[2048], 
                        hidden_dim=256, tgt_num_ch=512)
        self.deTrans_2_3 = transform(num_channels=[1024], 
                        hidden_dim=256, tgt_num_ch=512)
        self.deTrans_2_2 = transform(num_channels=[512], 
                        hidden_dim=256, tgt_num_ch=512)
        self.refine1x1_2_4 = ConvBlock1x1(256, 128)
        self.refine1x1_2_3 = ConvBlock1x1(256, 128)
        self.refine1x1_2_2 = ConvBlock1x1(256, 128)
        self.refine1x1_2 = ConvBlock1x1(512, 128)
        self.enhance_2 = ConvBlock3x3(128*4, 128)

        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            p = self.p[i]

            # # encoder不同层特征之间融合
            # self.deformableTransforms[("encoderFeature", i, 0)] = transform(num_channels=[self.num_ch_enc[-1 - i]], 
            #         hidden_dim=128, H=self.H[i], W=self.W[i], p=int(p), scale_factor=1,
            #         tgt_num_ch=self.num_ch_enc[i])

            # # decoder不同层特征之间融合
            # self.deformableTransforms[("decoderFeature", i, 0)] = transform(num_channels=[self.num_ch_enc[-1 - i] + 128], 
            #         hidden_dim=128, H=self.H[i], W=self.W[i], p=int(p), scale_factor=1,
            #         tgt_num_ch=num_ch_out)

            if i >= 5:
                self.deformableTransforms[("up", i, 0)] = transform(num_channels=[num_ch_in], 
                    hidden_dim=int(num_ch_out), H=self.H[i], W=self.W[i], p=int(p),
                    tgt_num_ch=self.num_ch_enc[i - 1])
            else:
                if i == 4:
                    # num_ch_in *= 2
                    # num_ch_in += 128
                    pass
                self.convs[("upconv", i, 0)] = ConvBlock3x3(num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # if i >= 5:
            #     num_ch_in += num_ch_in
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # concat上一次的深度结果
            if i+1 in self.scales:
                num_ch_in += 1
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock3x3(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.deformableTransforms2cuda = nn.ModuleList(list(self.deformableTransforms.values()))
        self.sigmoid = nn.Sigmoid()
    
    def semantic_enhance(self, features):
        out = features[:3]
        out[2] = torch.cat([self.refine1x1_2_4(self.deTrans_2_4([features[4]], out[2])), 
                            self.refine1x1_2_3(self.deTrans_2_3([features[3]], out[2])),
                            self.refine1x1_2_2(self.deTrans_2_2([features[2]], out[2])), 
                            self.refine1x1_2(out[2])], 1)
        out[2] = self.enhance_2(out[2])

        return out

    def forward(self, input_features):
        # # 对初始特征进行high level语义和low level细节的信息融合
        # input_features_t = []
        # for i in range(4, -1, -1):
        #     input_features_t.append(self.deformableTransforms[("encoderFeature", i, 0)]([input_features[-1 - i]], input_features[i]))

        # for i in range(4, -1, -1):
        #     # input_features[i] = input_features[i] + input_features_t[-1 - i]
        #     input_features[i] = torch.cat([input_features[i], input_features_t[-1 - i]], 1)

        # 增强中间层，得到3个特征
        input_features = self.semantic_enhance(input_features)

        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(2, -1, -1):
            if i >= 5:
                x = [self.deformableTransforms[("up", i, 0)]([x])]
            else:
                x = self.convs[("upconv", i, 0)](x)
                # x = torch.cat((x, self.deformableTransforms[("decoderFeature", i, 0)]([input_features[-1 - i]], x)), 1)
                x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            if i+1 in self.scales:
                x += [upsample(self.outputs[("disp", i+1)])]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
