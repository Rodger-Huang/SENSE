# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch.nn as nn
from functools import partial

from packnet_sfm.networks.depth.FPTResNet.resnet.resnet_encoder import ResnetEncoder
from packnet_sfm.networks.depth.FPTResNet.resnet.depth_decoder import DepthDecoder
from packnet_sfm.networks.depth.FPTResNet.resnet.layers import disp_to_depth

########################################################################################################################

class FPTResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.

    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, version=None, mode=None, **kwargs):
        super().__init__()
        assert version is not None, "DispResNet needs a version"

        num_layers = int(version[:2])       # First two characters are the number of layers
        pretrained = version[2:] == 'pt'    # If the last characters are "pt", use ImageNet pretraining
        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        self.scales = range(4)
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=self.scales, mode=mode)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, x, other_features=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x)
        x = self.decoder(x, other_features)
        disps = [x[('disp', i)] for i in self.scales]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]

########################################################################################################################
