# encoding: utf-8

import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from packnet_sfm.networks.depth.TreeFilter.config import config
from packnet_sfm.networks.depth.TreeFilter.base_model import resnet50
from packnet_sfm.networks.depth.TreeFilter.seg_opr.seg_oprs import (ConvBnRelu, RefineResidual)
from packnet_sfm.networks.depth.TreeFilter.base_model.resnet import BasicBlock
from packnet_sfm.networks.depth.TreeFilter.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from packnet_sfm.networks.depth.TreeFilter.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D
from packnet_sfm.networks.depth.TreeFilter.utils.init_func import init_weight

class TreeNetworkExtra(nn.Module):
    def __init__(self, inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(TreeNetworkExtra, self).__init__()
        business_channel_num = config.business_channel_num
        embed_channel_num = config.embed_channel_num

        self.backbone = resnet50(pretrained_model, inplace=inplace,
                                  norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        block_channel_nums = self.backbone.layer_channel_nums

        self.scales = range(4)

        self.latent_layers = nn.ModuleList()
        self.refine_layers = nn.ModuleList()
        self.embed_layers = nn.ModuleList()
        self.mst_layers = nn.ModuleList()
        self.tree_filter_layers = nn.ModuleList()
        self.predict_layers = nn.ModuleList()
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(block_channel_nums[-1], business_channel_num,
                       1, 1, 0, has_bn=True, has_relu=True, has_bias=False,
                       norm_layer=norm_layer))
        for idx, channel in enumerate(block_channel_nums[::-1]):
            self.latent_layers.append(
                RefineResidual(channel, business_channel_num, 3,
                               norm_layer=norm_layer, has_relu=True)
            )
            self.refine_layers.append(
                RefineResidual(business_channel_num, business_channel_num, 3,
                               norm_layer=norm_layer, has_relu=True)
            )
            self.embed_layers.append(
                ConvBnRelu(business_channel_num, embed_channel_num, 1, 1, 0, has_bn=False,
                           has_relu=False, has_bias=False, norm_layer=norm_layer))
            self.mst_layers.append(MinimumSpanningTree(TreeFilter2D.norm2_distance))
            self.tree_filter_layers.append(TreeFilter2D(groups=config.tree_filter_group_num))
            if idx-1 in self.scales:
                self.predict_layers.append(PredictHead(business_channel_num, 1, norm_layer=norm_layer))

        self.business_layers = [self.global_context, self.latent_layers, self.refine_layers,
                                self.predict_layers, self.embed_layers, self.mst_layers,
                                self.tree_filter_layers]

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

        init_weight(self.business_layers, nn.init.xavier_uniform_, # nn.init.kaiming_normal_,
                nn.BatchNorm2d, config.bn_eps, config.bn_momentum,)
                # mode='fan_in', nonlinearity='relu')

    def forward(self, data):
        disps = []

        blocks = self.backbone(data)
        blocks.reverse()

        gap = self.global_context(blocks[0])
        scale_factor = (blocks[0].shape[2], blocks[0].shape[3])
        last_fm = F.interpolate(gap, scale_factor=scale_factor,
                                mode='bilinear', align_corners=True)
        refined_fms = []
        for idx, (fm, latent_layer, refine_layer,
                  embed_layer, mst, tree_filter
                  ) in enumerate(
            zip(blocks,
                self.latent_layers,
                self.refine_layers,
                self.embed_layers,
                self.mst_layers,
                self.tree_filter_layers)):
            latent = latent_layer(fm)
            if last_fm is not None:
                tree = mst(fm)
                embed = embed_layer(last_fm)
                fusion = latent + tree_filter(last_fm, embed, tree)
                refined_fms.append(refine_layer(fusion))
            else:
                refined_fms.append(latent)
            last_fm = F.interpolate(refined_fms[-1], scale_factor=2, mode='bilinear',
                                    align_corners=True)

            if idx-1 in self.scales:
                disps.append(self.predict_layers[idx-1](last_fm))

        disps.reverse()

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]


class PredictHead(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(PredictHead, self).__init__()
        self.head_layers = nn.Sequential(
            RefineResidual(in_planes, in_planes, norm_layer=norm_layer, has_relu=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1,
                      stride=1, padding=0))

    def forward(self, x):
        x = self.head_layers(x)
        x = F.sigmoid(x)
        return x


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
