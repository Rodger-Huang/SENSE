import torch
from torch import nn
import torch.nn.functional as F

from .config import cfg as args

from .backbone import build_backbone
from .transformer import build_deforamble_transformer

from .util.misc import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid

from einops import rearrange

class DeformableDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_channels=[2048], num_feature_levels=4, 
        hidden_dim=256, H=6, W=20, p=2, scale_factor=1, tgt_num_ch=1024):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()
        self.transformer = transformer
        self.p = p
        self.H = H // self.p
        self.W = W // self.p
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        # self.query_embed = nn.Embedding(self.H*scale_factor * self.W*scale_factor, hidden_dim) # *2)
        # self.adapt_embed = MLP(hidden_dim, hidden_dim, hidden_dim * p, 3)
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                )])
            self.tgt_in_proj = nn.Sequential(
                    nn.Conv2d(tgt_num_ch, hidden_dim, kernel_size=1),
                    nn.GroupNorm(16, hidden_dim),
                )
            # self.tgt_out_proj = nn.Sequential(
            #         nn.Conv2d(hidden_dim, tgt_num_ch, kernel_size=1),
            #         nn.GroupNorm(16, tgt_num_ch),
            #     )
        self.backbone = backbone

        # nn.init.constant_(self.adapt_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.adapt_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        for proj in [self.tgt_in_proj]: # , self.tgt_out_proj]:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, x: NestedTensor, tgt):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = {}
        for name, sample in enumerate(x):
            if isinstance(sample, (list, torch.Tensor)):
                samples[name] = nested_tensor_from_tensor_list(sample)
        
        features, pos = self.backbone(samples)

        samples = {}
        for name, sample in enumerate([tgt]):
            if isinstance(sample, (list, torch.Tensor)):
                samples[name] = nested_tensor_from_tensor_list(sample)
        
        _, query_embed = self.backbone(samples)

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

        hs = self.transformer(srcs, masks, pos, query_embed[0], self.tgt_in_proj(tgt))
        
        B, C, H, W = tgt.shape
        out = hs.permute(0, 2, 1).view(B, self.hidden_dim, H*self.scale_factor, W*self.scale_factor)
        # out = self.tgt_out_proj(out.contiguous())
        return out.contiguous()


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

def transform(num_channels=[2048], hidden_dim=256, H=6, W=20, 
        p=2, scale_factor=1, tgt_num_ch=1024):
    p = 1
    hidden_dim = hidden_dim * p
    backbone = build_backbone(args, hidden_dim)

    transformer = build_deforamble_transformer(hidden_dim, args)

    model = DeformableDETR(
        backbone,
        transformer,
        num_channels=num_channels,
        num_feature_levels=args.num_feature_levels,
        hidden_dim=hidden_dim,
        H=H, W=W, p=p, scale_factor=scale_factor,
        tgt_num_ch=tgt_num_ch,
    )

    return model
