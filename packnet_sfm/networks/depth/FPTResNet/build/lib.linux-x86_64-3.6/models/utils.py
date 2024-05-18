import torch.nn as nn


class v_transform(nn.Conv1d):
    """Conv1d for v_transform"""

class qk_transform(nn.Conv1d):
    """Conv1d for qk_transform"""

class kv_transform(nn.Conv1d):
    """Conv1d for kv_transform"""

class q_transform(nn.Conv1d):
    """Conv1d for q_transform"""
