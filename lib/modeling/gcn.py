import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph():
    def __init__(self, dataset, max_hop=3, dilation=1, node_set=17):
        self.dataset = dataset.split('-')[0]
        self.max_hop = max_hop
        self.dilation = dilation
        self.node_set = node_set
        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            parts = [
                np.array([5, 6, 7]),              # left_arm
                np.array([2, 3, 4]),              # right_arm
                np.array([11, 12, 13]),           # left_leg
                np.array([8, 9, 10]),             # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
        elif self.dataset == 'ntu':
            num_node = 25
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
            parts = [
                np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
                np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4, 21]) - 1          # torso
            ]
        elif self.dataset == 'sysu':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,     # left_arm
                np.array([9, 10, 11, 12]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4]) - 1          # torso
            ]
        elif self.dataset == 'ucla':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (3, 5), (5, 6),
                              (6, 7), (7, 8), (3, 9), (9, 10), (10, 11),
                              (11, 12), (1, 13), (13, 14), (14, 15), (15, 16),
                              (1, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([2,2,2,3,3,5,6,7,3,9,10,11,1,13,14,15,1,17,18,19]) - 1
            parts = [
                np.array([5, 6, 7, 8]) - 1,     # left_arm
                np.array([9, 10, 11, 12]) - 1,  # right_arm
                np.array([13, 14, 15, 16]) - 1,         # left_leg
                np.array([17, 18, 19, 20]) - 1,         # right_leg
                np.array([1, 2, 3, 4]) - 1          # torso
            ]
        elif self.dataset == 'cmu':
            num_node = 26
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (12, 13), (13, 14), (12, 15), (15, 16),
                              (16, 17), (17, 18), (18, 19), (17, 20), (12, 21),
                              (21, 22), (22, 23), (23, 24), (24, 25), (23, 26)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,10,10,10,11,12,13,12,15,16,17,18,17,12,21,22,23,24,23]) - 1
            parts = [
                np.array([15, 16, 17, 18, 19, 20]) - 1,     # left_arm
                np.array([21, 22, 23, 24, 25, 26]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,         # left_leg
                np.array([5, 6, 7, 8]) - 1,         # right_leg
                np.array([9, 10, 11, 12, 13, 14]) - 1          # torso
            ]
        elif self.dataset == 'h36m':
            num_node = 20
            neighbor_1base = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7),
                              (7, 8), (1, 9), (5, 9), (9, 10), (10, 11),
                              (11, 12), (10, 13), (13, 14), (14, 15), (15, 16),
                              (10, 17), (17, 18), (18, 19), (19, 20)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            connect_joint = np.array([9,1,2,3,9,5,6,7,9,9,10,11,10,13,14,15,10,17,18,19]) - 1
            parts = [
                np.array([13, 14, 15, 16]) - 1,     # left_arm
                np.array([17, 18, 19, 20]) - 1,  # right_arm
                np.array([1, 2, 3, 4]) - 1,         # left_leg
                np.array([5, 6, 7, 8]) - 1,         # right_leg
                np.array([9, 10, 11, 12]) - 1          # torso
            ]
        elif self.dataset == 'coco':
            # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",           
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
            num_node = 17
            self_link = [(i, i) for i in range(num_node)]
            neighbor_link = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6),
                             (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                             (11, 13), (13, 15), (12, 14), (14, 16)]
            self.edge = self_link + neighbor_link
            self.center = 0
            connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            parts = [
                np.array([5, 7, 9]),       # left_arm
                np.array([6, 8, 10]),      # right_arm
                np.array([11, 13, 15]),    # left_leg
                np.array([12, 14, 16]),    # right_leg
                np.array([5, 6, 11, 12, 0, 1, 2, 3, 4]),  # torso + head
            ]
        elif self.dataset == 'all':
            num_node = self.node_set
            neighbor_link = []
            for i in range(num_node):
                for j in range(num_node):
                    neighbor_link.append((i, j))
            self.edge = neighbor_link
            parts = []
            connect_joint = []
        else:
            num_node, neighbor_link, connect_joint, parts = 0, [], [], []
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        if self.dataset == 'all':
            edge = neighbor_link
        else:
            edge = self_link + neighbor_link

        return num_node, edge, connect_joint, parts

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[hop_dis == hop] = 1
        normalize_adjacency = self._normalize_digraph(adjacency)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]
        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD


class Spatial_Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual=True, **kwargs):
        super(Spatial_Basic_Block, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, kernel_size)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res_block = self.residual(x)

        x, A = self.conv(x, A)
        # x = self.bn(x)
        x = self.relu(x + res_block)

        return x, A

# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = kernel_size
        # weights of different spatial classes
        # self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)
        self.gcn = nn.Conv2d(
            in_channels,
            out_channels*self.s_kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            dilation=(1, 1),
            bias=True)

    def forward(self, x, A):
        assert A.size(0) == self.s_kernel_size

        # numbers in same class have same weight
        x = self.gcn(x)
        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        # spatial graph convolution

        x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        return x.contiguous(), A
    

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3):
        super(GraphConv, self).__init__()

        self.s_kernel_size = kernel_size
        self.dropout = dropout
        self.gcn = nn.Conv2d(
            in_channels,
            out_channels*self.s_kernel_size,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            dilation=(1, 1),
            bias=True)

    def forward(self, x, A):
        x = torch.einsum('nctv,vw->nctw', (x, A)).contiguous()
        x = self.gcn(x)
        x = F.dropout(x, self.dropout)
        return x.contiguous(), A


def nconv(x, A):
    return torch.einsum('bcnt,nm->bcmt', (x, A)).contiguous()


class Diffusion_GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.3, support_len=1, order=1):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.conv = nn.Conv2d(c_in, c_out, (1, 1), padding=(
            0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order
    def forward(self, x, a):
        out = [x]
        x1 = nconv(x, a)
        out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.conv(h)
        h = F.dropout(h, self.dropout)
        return h, a 
    

