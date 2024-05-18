import torch
from torch import nn
import torch.nn.functional as F

from .config import cfg as args

from .backbone import build_backbone
from .transformer import build_transformer

from .util.misc import NestedTensor, nested_tensor_from_tensor_list

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

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries):
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
        self.query_embed = nn.Embedding(12*40, 256)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.input_proj = nn.Conv2d(64, 4, kernel_size=1)
        self.backbone = backbone

        out_channels = 1
        self.disp4_layer = InvDepth(256, out_channels=out_channels)
        self.disp3_layer = InvDepth(256, out_channels=out_channels)
        self.disp2_layer = InvDepth(256, out_channels=out_channels)
        self.disp1_layer = InvDepth(256, out_channels=out_channels)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        decoder_out = hs[0]
        encoder_out = hs[1]

        disp4 = self.disp4_layer(decoder_out[-4])
        disp3 = self.disp3_layer(decoder_out[-3])
        disp2 = self.disp2_layer(decoder_out[-2])
        disp1 = self.disp1_layer(decoder_out[-1])

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

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
    )

    return model
