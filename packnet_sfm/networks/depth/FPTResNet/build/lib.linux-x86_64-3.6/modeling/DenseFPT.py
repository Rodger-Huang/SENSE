# ---------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from packnet_sfm.networks.depth.FPTResNet.modeling.self_trans import SelfTrans
from packnet_sfm.networks.depth.FPTResNet.modeling.rendering_trans import RenderTrans
from packnet_sfm.networks.depth.FPTResNet.modeling.grounding_trans import GroundTrans
import packnet_sfm.networks.depth.FPTResNet.nn as mynn


class DenseFPT(nn.Module):
    def __init__(self, feature_dim, with_norm='none', upsample_method='bilinear', num_feature_maps=4, channels=[]):
        super(DenseFPT, self).__init__()
        self.num_feature_maps = num_feature_maps
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']
        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method, align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm
        
        self.st_p = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        
        gt_p_list = []
        for i in range(num_feature_maps - 1):
            gt_p_list.append(GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True))
        self.gt_p = nn.ModuleList(gt_p_list)
        
        if with_norm != 'none':
            fpn_p_1x1 = []
            for i in range(num_feature_maps):
                fpn_p_1x1.append(nn.Sequential(*[nn.Conv2d(channels[i], feature_dim, 1, bias=False), norm(feature_dim)]))
            self.fpn_p_1x1 = nn.ModuleList(fpn_p_1x1)
            
            self.fpt_p = nn.Sequential(*[nn.Conv2d(feature_dim*num_feature_maps, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            fpn_p_1x1 = []
            for i in range(num_feature_maps):
                fpn_p_1x1.append(nn.Conv2d(channels[i], feature_dim, 1))
            self.fpn_p_1x1 = nn.ModuleList(fpn_p_1x1)
            
            self.fpt_p = nn.Conv2d(feature_dim*(num_feature_maps+1), feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, features):
        fpn_p_1 = []
        for i in range(self.num_feature_maps):
            fpn_p_1.append(self.fpn_p_1x1[i](features[i]))

        fpt_p_out = [self.st_p(fpn_p_1[-1])]
        for i in range(self.num_feature_maps - 1):
            fpt_p_out += [self.gt_p[i](fpn_p_1[-1], fpn_p_1[i])]
        fpt_p_out += [fpn_p_1[-1]]
        fpt_p_out = torch.cat(fpt_p_out, 1)
        
        fpt_p = self.fpt_p(fpt_p_out)

        return fpt_p
