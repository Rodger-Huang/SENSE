import torch
from torch import nn
import torch.distributed as dist

from ..functions.mst import mst
from ..functions.bfs import bfs
from ..functions.refine import refine

class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func
    
    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)

        # slope_index = torch.stack([raw_index[:-1, :-1], raw_index[1:, 1:]], 2)

        index = torch.cat([row_index.reshape(1, -1, 2), 
                           col_index.reshape(1, -1, 2),
                           ], 1) # slope_index.reshape(1, -1, 2)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])

        # weight_slope = self.distance_func(fm[:, :, :-1, :-1], fm[:, :, 1:, 1:])

        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])

        # weight_slope = weight_slope.reshape([batch, -1])

        weight = torch.cat([weight_row, weight_col], dim=1) # , weight_slope
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def forward(self, guide_in):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in) # (B, (H-1)*W + H*(W-1), 2)
            weight = self._build_feature_weight(guide_in) # (B, (H-1)*W + H*(W-1))
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree


class TreeFilter2D(nn.Module):
    def __init__(self, groups=1, distance_func=None,
                 mapping_func=torch.exp):
        super(TreeFilter2D, self).__init__()
        self.groups = groups
        self.mapping_func = mapping_func
        if distance_func is None:
            self.distance_func = self.norm2_distance
        else:
            self.distance_func = distance_func
        
        # self.distance_weight = nn.Parameter(torch.ones(1), requires_grad=True)

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     nn.init.constant_(self.distance_weight.weight.data, 1.)
    #     nn.init.constant_(self.distance_weight.bias.data, 0.)

    @staticmethod
    def norm2_distance(fm_ref, fm_tar):
        diff = fm_ref - fm_tar
        weight = (diff * diff).sum(dim=1)
        # weight = diff.abs().sum(dim=1)
        return weight

    @staticmethod
    def batch_index_opr(data, index):
        with torch.no_grad():
            channel = data.shape[1]
            index = index.unsqueeze(1).expand(-1, channel, -1).long()
        data = torch.gather(data, 2, index)
        return data

    def build_edge_weight(self, fm, sorted_index, sorted_parent):
        batch   = fm.shape[0]
        channel = fm.shape[1]
        vertex  = fm.shape[2] * fm.shape[3]
    
        fm = fm.reshape([batch, channel, -1])
        fm_source = self.batch_index_opr(fm, sorted_index)
        fm_target = self.batch_index_opr(fm_source, sorted_parent)
        fm_source = fm_source.reshape([-1, channel // self.groups, vertex])
        fm_target = fm_target.reshape([-1, channel // self.groups, vertex])
    
        edge_weight = self.distance_func(fm_source, fm_target)
        
        # # 对权重进行一个减小处理
        # edge_weight = 0.5 * edge_weight / torch.mean(edge_weight, dim=1, keepdim=True)
        # edge_weight = self.distance_weight * edge_weight

        edge_weight = torch.exp(-edge_weight)

        # edge_weight_new = edge_weight.clone()
        # edge_weight_new[edge_weight < edge_weight.mean()] *= 0.5

        return edge_weight
        
    def build_self_weight(self, fm, sorted_index):
        batch   = fm.shape[0]
        channel = fm.shape[1]
        vertex  = fm.shape[2] * fm.shape[3]

        fm = fm.reshape(-1, fm.shape[1] // self.groups, vertex)
        self_dist = self.distance_func(fm, 0)
        self_weight = self.mapping_func(-self_dist)
        att_weight = self_weight.reshape(-1, self.groups, vertex)
        att_weight = self.batch_index_opr(att_weight, sorted_index)
        att_weight = att_weight.reshape(-1, vertex)
        return self_weight, att_weight

    def split_group(self, feature_in, *tree_orders):
        feature_in = feature_in.reshape(feature_in.shape[0] * self.groups, 
                                        feature_in.shape[1] // self.groups,
                                        -1)
        returns = [feature_in.contiguous()]
        for order in tree_orders:
            order = order.unsqueeze(1).expand(order.shape[0], self.groups, *order.shape[1:])
            order = order.reshape(-1, *order.shape[2:])
            returns.append(order.contiguous())
        return tuple(returns)

    def forward(self, feature_in, embed_in, tree, guide_in=None, self_dist_in=None):
        ori_shape = feature_in.shape
        sorted_index, sorted_parent, sorted_child = bfs(tree, 4) # 4
        edge_weight = self.build_edge_weight(embed_in, sorted_index, sorted_parent)
        if self_dist_in is None:
            self_weight = torch.ones_like(edge_weight)
        else:
            self_weight, att_weight = self.build_self_weight(self_dist_in, sorted_index)
            edge_weight = edge_weight * att_weight

        if guide_in is not None:
            guide_weight = self.build_edge_weight(guide_in, sorted_index, sorted_parent)
            edge_weight = edge_weight * guide_weight
            
        feature_in, sorted_index, sorted_parent, sorted_child = \
            self.split_group(feature_in, sorted_index, sorted_parent, sorted_child)
        feature_out = refine(feature_in, edge_weight, sorted_index, 
                             sorted_parent, sorted_child)
        feature_out = feature_out.reshape(ori_shape)
        return feature_out

