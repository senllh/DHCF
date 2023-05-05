import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


# Make initial feature dim to embedding dim
class DimReduceLayer(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DimReduceLayer, self).__init__()

        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feat):
        return torch.mm(feat, self.weight) + self.bias


# Feature disentangle layer
class RoutingLayer(nn.Module):
    def __init__(self, k, routit, dim, tau):
        super(RoutingLayer, self).__init__()
        self.k = k
        self.routit = routit
        self.dim = dim
        self.tau = tau

    def forward(self, x, edge_index):
        m, src, trg = edge_index.shape[1], edge_index[0], edge_index[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k

        x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)  # neighbors' feature  # 它这里是把一个特征直接划分成了 k 份
        c = x  # node-neighbor attention aspect factor

        for t in range(self.routit):
            # z [18092, k, dim/k], 是 source节点的表示,  c[trg] [18092, k, dim/k]是尾节点的表示
            # p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)  #  元素乘积
            p = torch.matmul(z.view(m, k, delta_d), c[trg].view(m, delta_d, k)).sum(dim=2)  # dot
            # p = torch.mean(torch.stack([z * c[trg].view(m, k, delta_d)], dim=3), dim=3).sum(dim=2)
            p = F.softmax(p / self.tau, dim=1)  # 计算不同 通道的 权重
            p = p.view(-1, 1).repeat(1, delta_d).view(m, k, delta_d)  # [18092, k, dim/k]

            weight_sum = (p * z).view(m, d)  # weight sum (node attention * neighbors feature)
            # c = c.index_add_(0, trg, weight_sum)  # c [6072, 64] ; trg [12082] ;weight_sum [12082, 64]
            c = torch.zeros(n, d, device=x.device)  # c 是否迭代利用
            c = c.index_add_(0, trg, weight_sum)
            c += x
            if t < self.routit - 1:
                c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)  # embedding normalize aspect factor
        return c


# # Feature disentangle layer
# class RoutingLayer1(nn.Module):
#     def __init__(self, k, routit, dim, tau):
#         super(RoutingLayer1, self).__init__()
#         self.k = k
#         self.routit = routit
#         self.dim = dim
#         self.tau = tau
#
#     def forward(self, x, edge_index):
#         m, src, trg = edge_index.shape[1], edge_index[0], edge_index[1]
#         n, d = x.shape
#         k, delta_d = self.k, d // self.k
#
#         x = F.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
#         z = x[trg].view(m, k, delta_d)  # neighbors' feature  # 它这里是把一个特征直接划分成了 k 份
#         c = x
#         scatter_idx = trg.view(m, 1).expand(m, d)
#         for clus_iter in range(self.routit):
#             # p = (z * c[trg].view(m, k, delta_d)).sum(dim=2)
#             p = torch.matmul(z.view(m, k, delta_d), c[trg].view(m, delta_d, k)).sum(dim=2)
#             p = F.softmax(p / self.tau, dim=1)
#             scatter_src = (z * p.view(m, k, 1)).view(m, d)
#             c = torch.zeros(n, d, device=x.device)  # c 是否迭代利用
#             c.scatter_add_(0, scatter_idx, scatter_src)
#             c += x
#             # noinspection PyArgumentList
#             if clus_iter < self.routit - 1:
#                 c = F.normalize(c.view(n, k, delta_d), dim=2).view(n, d)
#         return c

class DisenGCN(nn.Module):
    def __init__(self, in_dim, n_hid, n_layer, K, routit, cor_weight, tau):
        super(DisenGCN, self).__init__()
        self.K = K
        self.outdim = n_hid
        self.routit = routit
        self.cor_weight = cor_weight
        self.tau = tau
        if n_hid % self.K == 0:
            n_hid = n_hid
        else:
            # n_hid = n_hid
            n_hid = self.lcm(n_hid, self.K)
        conv_ls = []
        self.BN = nn.ModuleList()
        for i in range(n_layer):
            conv = RoutingLayer(k=self.K, routit=self.routit, dim=n_hid, tau=self.tau)  # 插入路由层
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
            self.BN.append(nn.BatchNorm1d(n_hid))
        self.lin = nn.Linear(in_dim, n_hid)
        self.conv_ls = conv_ls  # 多层 解耦GCN
        self.mlp = nn.Linear(n_hid, self.outdim)
        self.Cor_loss = Cor_loss(self.cor_weight, self.K, n_hid)

    def lcm(self, x, y):  # 最小公倍数
        #  获取最大的数
        if x > y:
            greater = x
        else:
            greater = y
        while (True):
            if ((greater % x == 0) and (greater % y == 0)):
                lcm = greater
                break
            greater += 1
        return lcm

    def forward(self, feat, src_trg_edges):
        '''
        :param feat: [n_node.dim]
        :param src_trg_edges: adj 的稀疏表示
        :return:
        '''
        x = self.lin(feat)
        for conv, bn in zip(self.conv_ls, self.BN):
            x = conv(x, src_trg_edges)
            # x = F.leaky_relu(x)  # 可以试试去掉激活函数
        # loss = self.Cor_loss(x)
        out = self.mlp(x)
        return out, x  # F.log_softmax(x, dim=1)


class Cor_loss(nn.Module):
    def __init__(self, cor_weight, channels, hidden_size):
        super(Cor_loss, self).__init__()
        self.channel_num = channels
        self.cor_weight =cor_weight
        self.hidden_size = hidden_size

    def forward(self, embedding):#解耦任意factor对
        if self.cor_weight==0:
            return 0
        else:
            embedding = embedding.view(-1, self.hidden_size)
            embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
            cor_loss = torch.tensor(0, dtype = torch.float)
            for i in range(self.channel_num):
                for j in range(i+1, self.channel_num):
                    x=embedding_weight[i]
                    y=embedding_weight[j]
                    cor_loss = cor_loss+self._create_distance_correlation(x, y)
            b= (self.channel_num+1.0) * self.channel_num/2
            cor_loss = self.cor_weight * torch.div(cor_loss,b)
        return cor_loss

    # def forward_(self, embedding):#解耦相邻factor对（内存不足情况下的次优解）
    #     if self.cor_weight==0:
    #         return 0
    #     else:
    #         embedding = embedding.view(-1, self.hidden_size)
    #         embedding_weight = torch.chunk(embedding, self.channel_num, dim=1)
    #         cor_loss = torch.tensor(0, dtype = torch.float)
    #         for i in range(self.channel_num-1):
    #             x=embedding_weight[i]
    #             y=embedding_weight[i+1]
    #             cor_loss = cor_loss+self._create_distance_correlation(x, y)
    #         b= (self.channel_num+1.0)* self.channel_num/2
    #         cor_loss = self.cor_weight * torch.div(cor_loss,b)
    #     return cor_loss

    def _create_distance_correlation(self, x, y):
        zero = torch.tensor(0, dtype=torch.float).cuda()
        def _create_centered_distance(X, zero):
            r = torch.sum(torch.square(X), 1, keepdim=True)
            X_t = torch.transpose(X, 1, 0)
            r_t = torch.transpose(r, 1, 0)
            D = torch.sqrt(torch.maximum(r-2*torch.matmul(X,X_t)+r_t,zero)+1e-8)
            D = D - torch.mean(D, dim=0, keepdim=True)-torch.mean(D,dim=1,keepdim=True)+torch.mean(D)
            return D

        def _create_distance_covariance(D1,D2,zero):
                n_samples = D1.shape[0]
                n_samples = torch.tensor(n_samples, dtype=torch.float)
                sum = torch.sum(D1*D2)
                sum = torch.div(sum,n_samples*n_samples)
                dcov=torch.sqrt(torch.maximum(sum,zero)+1e-8)
                return dcov

        D1 = _create_centered_distance(x,zero)
        D2 = _create_centered_distance(y,zero)

        dcov_12 = _create_distance_covariance(D1, D2,zero)
        dcov_11 = _create_distance_covariance(D1, D1,zero)
        dcov_22 = _create_distance_covariance(D2, D2,zero)

        dcor = torch.sqrt(torch.maximum(dcov_11 * dcov_22, zero))+1e-10
        dcor = torch.div(dcov_12,dcor)
        return dcor
