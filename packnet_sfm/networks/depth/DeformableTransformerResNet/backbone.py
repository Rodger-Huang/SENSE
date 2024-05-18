# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from torch import nn
from typing import Dict, List

from .util.misc import NestedTensor

from .position_encoding import build_position_encoding


class Joiner(nn.Sequential):
    def __init__(self, position_embedding):
        super().__init__(position_embedding)
        self.strides = [1]

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[0](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args, hidden_dim):
    position_embedding = build_position_encoding(args, hidden_dim)
    model = Joiner(position_embedding)
    return model
