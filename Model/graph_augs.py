import numpy as np
from IPython import embed
import copy
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx

import torch
import torch.nn as nn

from torch_geometric.utils import subgraph, k_hop_subgraph

def NodeDrop(data, aug_ratio):
    data = copy.deepcopy(data)
    x = data.x#[:, :300]
    edge_index = data.edge_index

    # 随机dropout
    drop_num = int(data.num_nodes * aug_ratio)
    keep_num = data.num_nodes - drop_num
    keep_idx = torch.randperm(data.num_nodes)[:keep_num]
    edge_index, _ = subgraph(keep_idx, edge_index)
    drop_idx = torch.ones(x.shape[0], dtype=bool)
    drop_idx[keep_idx] = False
    # root 不 dropout
    drop_idx[data.root_index] = False
    x[drop_idx] = 0
    data.x = x
    data.edge_index = edge_index
    return data

def EdgePerturb(data, aug_ratio):
    data = copy.deepcopy(data)
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index

    unif = torch.ones(2, node_num)
    add_edge_idx = unif.multinomial(permute_num, replacement=True).to(data.x.device)

    # # 随机抽样
    unif = torch.ones(edge_num)
    keep_edge_idx = unif.multinomial((edge_num - permute_num), replacement=False)

    # edge_index = edge_index[:, keep_edge_idx]
    #
    edge_index = torch.cat((edge_index[:, keep_edge_idx], add_edge_idx), dim=1)
    data.edge_index = edge_index
    return data

def AttrMask(data, aug_ratio):
    data = copy.deepcopy(data)
    # 随机 mask
    mask_num = int(data.num_nodes * aug_ratio)
    unif = torch.ones(data.num_nodes)
    unif[data.root_index] = 0
    mask_idx = unif.multinomial(mask_num, replacement=False)
    token = data.x.mean(dim=0)
    # shape = data.x.mean(dim=0).shape
    # token = torch.rand(shape[0]).cuda()
    data.x[mask_idx] = token
    return data

class Graph_Augmentor(nn.Module):
    def __init__(self, aug_ratio, preset=-1):
        super().__init__()
        self.aug_ratio = aug_ratio
        self.aug = preset
    
    def forward(self, data):
        data1 = data
        data = copy.deepcopy(data)
        if self.aug_ratio > 0:
            self.aug = 1#np.random.randint(3)
            if self.aug == 0:
                # print("node drop")
                data = NodeDrop(data, self.aug_ratio)
            elif self.aug == 1:
                # print("edge perturb")
                data = EdgePerturb(data, self.aug_ratio)
            elif self.aug == 2:
                # print("attr mask")
                data = AttrMask(data, self.aug_ratio)
            elif self.aug == 3:
                # print("attr mask")
                data = AttrMask(data, self.aug_ratio)
                data = NodeDrop(data, self.aug_ratio)
                data = EdgePerturb(data, self.aug_ratio)
            else:
                print('sample augmentation error')
                assert False
        return data