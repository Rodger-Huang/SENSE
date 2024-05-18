# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from .default import _C as cfg

# from packnet_sfm.networks.depth.HRNet.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
# from packnet_sfm.networks.depth.HRNet.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D
# from packnet_sfm.networks.depth.HRNet.net_canny import CannyNet
# from packnet_sfm.networks.depth.HRNet.kernels.lib_tree_filter.tree_filter_v2 import TreeFilterV2

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(192, num_pos_feats)
        self.col_embed = nn.Embedding(640, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, use_tree=True,
                 num_channels_pre_layer=None, num_channels_cur_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        self.transition_layers = self._make_transition_layer(num_channels_cur_layer)

        if use_tree:
            self.tree_filter_layers = self._tree_layers()

        self.use_tree = use_tree

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        # for i in range(num_branches if self.multi_scale_output else 1):
        i = 0
        fuse_layer = []
        for j in range(num_branches):
            if j > i:
                fuse_layer.append(nn.Sequential(
                    nn.Conv2d(num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False),
                    BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
            elif j == i:
                fuse_layer.append(None)
            else:
                conv3x3s = []
                for k in range(i-j):
                    if k == i - j - 1:
                        num_outchannels_conv3x3 = num_inchannels[i]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False),
                            BatchNorm2d(num_outchannels_conv3x3, 
                                        momentum=BN_MOMENTUM)))
                    else:
                        num_outchannels_conv3x3 = num_inchannels[j]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False),
                            BatchNorm2d(num_outchannels_conv3x3,
                                        momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                fuse_layer.append(nn.Sequential(*conv3x3s))
        fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def _tree_layers(self):
        if self.num_branches == 1:
            return None, None, None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        tree_filter_layers = []
        for i in range(num_branches):
            tree_filter_layer = []
            # for j in range(num_branches):
            tree_filter_layer.append(TreeFilterV2(guide_channels=num_inchannels[i],
                in_channels=num_inchannels[i], embed_channels=num_inchannels[i],
                num_groups=16))

            tree_filter_layers.append(nn.ModuleList(tree_filter_layer))

        return nn.ModuleList(tree_filter_layers)

    def _make_transition_layer(
            self, num_channels_cur_layer):

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        transition_layers = []
        for i in range(num_branches):
            transition_layer = []
            for j in range(num_branches):
                if j == 0:
                    transition_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(j):
                        inchannels = num_channels_cur_layer[k]
                        outchannels = num_channels_cur_layer[k+1]
                        conv3x3s.append(nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False),
                            BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                    transition_layer.append(nn.Sequential(*conv3x3s))

            transition_layers.append(nn.ModuleList(transition_layer))

        return nn.ModuleList(transition_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        y = x[0]
        for j in range(1, self.num_branches):
            width_output = x[0].shape[-1]
            height_output = x[0].shape[-2]
            
            y = y + F.interpolate(
                self.fuse_layers[0][j](x[j]),
                size=[height_output, width_output],
                mode='bilinear')

        if self.use_tree:
            fm = x[i]
            y = y + self.tree_filter_layers[i][0](fm, y)
        
        for j in range(self.num_branches):
            if self.transition_layers[0][j] is not None:
                x_fuse.append(self.transition_layers[0][j](x_fuse[0]))
            else:
                x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):
    def __init__(self, config, scales, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        self.scales = scales

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, use_tree=False,
            num_channels_pre_layer=[stage1_out_channel], num_channels_cur_layer=num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        # self.transition2 = self._make_transition_layer(
        #     [48, 96, 192], num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, use_tree=False,
            num_channels_pre_layer=pre_stage_channels, num_channels_cur_layer=num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True,
            use_tree=False, num_channels_pre_layer=pre_stage_channels, num_channels_cur_layer=num_channels)

        # pre_stage_channels = [48, 96, 192]
        # self.transition4 = self._make_transition_layer(
        #     pre_stage_channels, pre_stage_channels)
        
        self.last_layer = nn.ModuleList()
        # self.last_embed = nn.ModuleList()
        # self.last_mst = nn.ModuleList()
        # self.last_tree_filter = nn.ModuleList()
        # self.edge_layer = nn.ModuleList()
        for i in range(self.scales):
            self.last_layer.append(nn.Sequential(
                nn.Conv2d(
                    in_channels=pre_stage_channels[i],
                    out_channels=pre_stage_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=0),
                BatchNorm2d(pre_stage_channels[i], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=pre_stage_channels[i],
                    out_channels=1,
                    kernel_size=extra.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
                # nn.Sigmoid() if i == (self.scales - 1) else nn.Tanh(),
            ))
            # self.last_embed.append(nn.Sequential(
            #     nn.Conv2d(pre_stage_channels[i],
            #                 pre_stage_channels[i],
            #                 1,
            #                 1,
            #                 0,
            #                 bias=False),
            #     ))
            # self.edge_layer.append(nn.Sequential(
            #     nn.Conv2d(pre_stage_channels[i],
            #               pre_stage_channels[i],
            #               1,
            #               1,
            #               0,),
            #     BatchNorm2d(pre_stage_channels[i], momentum=BN_MOMENTUM),
            #     nn.ReLU(inplace=True),
            # ))
            # self.last_tree_filter.append(TreeFilter2D(groups=16))
            # self.last_mst.append(MinimumSpanningTree(TreeFilter2D.norm2_distance))
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)
        # self.gamma0 = nn.Parameter(torch.zeros(1))
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))
        # self.gamma3 = nn.Parameter(torch.zeros(1))
        self.init_weights(cfg.MODEL.PRETRAINED)
        # self.canny = CannyNet(threshold=3.0)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur=len(num_channels_cur_layer)
        num_branches_pre=len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            # if i < num_branches_pre:
            if i == 0:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i):
                    inchannels = num_channels_pre_layer[j]
                    outchannels = num_channels_cur_layer[j+1]
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, use_tree=True, 
                    num_channels_pre_layer=None, num_channels_cur_layer=None):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output,
                                      use_tree,
                                      num_channels_pre_layer, 
                                      num_channels_cur_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = (x - 0.45) / 0.225
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[0]))
            else:
                x_list.append(y_list[0])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[0]))
            else:
                x_list.append(y_list[0])
        x = self.stage4(x_list)

        x_dict = {}
        # Upsampling
        for i in range(self.scales):
            x_dict[i] = F.upsample(x[i], scale_factor=4, mode='bilinear')
        
        disps = []
        upsample_x = 0.
        for i in range(self.scales-1, -1, -1):
            disps.append(F.sigmoid(self.last_layer[i](x_dict[i]) + upsample_x))
            if i != 0:
                upsample_x = F.upsample(disps[-1], scale_factor=2, mode='bilinear')
        
        if self.training:
            return [self.scale_inv_depth(d)[0] for d in reversed(disps)]
        else:
            return self.scale_inv_depth(disps[-1])[0]


    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        # nn.init.constant_(self.gamma0, 0)
        # nn.init.constant_(self.gamma1, 0)
        # nn.init.constant_(self.gamma2, 0)
        # nn.init.constant_(self.gamma3, 0)

def seg_hrnet(scales, **kwargs):
    cfg.merge_from_file('packnet_sfm/networks/depth/HRNet/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
    model = HighResolutionNet(cfg, scales, **kwargs)

    return model

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
