import torch
from torch import nn
import torch.nn.functional as F

from .config import cfg as args

from .backbone import build_backbone
from .transformer import build_deforamble_transformer

from .util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid

from einops import rearrange

class InvDepth(nn.Module):
    """Inverse depth layer"""
    def __init__(self, in_channels, out_channels=1, min_depth=0.5):
        """
        Initializes an InvDepth object.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        min_depth : float
            Minimum depth value to calculate
        """
        super().__init__()
        self.min_depth = min_depth
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.pad = nn.ConstantPad2d([1] * 4, value=0)
        self.activ = nn.Sigmoid()

    def forward(self, x):
        """Runs the InvDepth layer."""
        x = self.conv1(self.pad(x))
        return self.activ(x) / self.min_depth

class DeformableDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 16384, 3)
        # num_queries=100,hidden_dim=256
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.H = 192 // 16
        self.W = 640 // 16
        query_embed = []
        for _ in range(256):
            query_embed.append(nn.Embedding(self.H*self.W, 8*2))
        self.query_embeds = nn.ModuleList(query_embed)
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.output_proj = nn.Conv2d(hidden_dim, backbone.num_channels, kernel_size=1)
        # self.input_proj = nn.Conv2d(64, 4, kernel_size=1)
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        out_channels = 1
        self.disp4_layer = InvDepth(8, out_channels=out_channels)
        self.disp3_layer = InvDepth(8, out_channels=out_channels)
        self.disp2_layer = InvDepth(8, out_channels=out_channels)
        self.disp1_layer = InvDepth(8, out_channels=out_channels)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs = self.transformer(srcs, masks, pos, self.query_embeds)

        out = hs
        
        # out = []
        # for i in range(4):
        #     out.append(F.interpolate(hs_out[i], size=[192, 640], mode='bilinear'))
        # out = hs_out

        # p = 16
        # out = rearrange(hs, 'm b (h w) (p1 p2 c) -> m b c (h p1) (w p2)', h = 12, w = 40, p1 = p, p2 = p)

        disp4 = self.disp4_layer(out[-4])
        disp3 = self.disp3_layer(out[-3])
        disp2 = self.disp2_layer(out[-2])
        disp1 = self.disp1_layer(out[-1])

        # disp1 = self.disp1_layer(encoder_out)
        # disp2 = disp1
        # disp3 = disp1
        # disp4 = disp1
        
        if self.training:
            return [disp1, disp2, disp3, disp4]
        else:
            return disp1


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def transform():
    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    model = DeformableDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
    )

    return model
