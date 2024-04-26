import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .st_gcn import st_gcn_SpGraph

class static_module(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # [temporal kernel, sptial kernel] 
                 num_nodes=16
                 ):
        super().__init__()
        self.p = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size[1], 1, 0, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels)
        )
        self.t = nn.Sequential(
            nn.Conv1d(in_channels*num_nodes, out_channels*num_nodes, kernel_size[0], 1, 1, groups=num_nodes, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels * num_nodes)
        )
    def forward(self, x):
           p, n, c, s = x.size()
           x = self.p(x.permute(1,3,2,0).contiguous().view(n*s, c, p)).view(n, s, c, p)
           x = self.t(x.permute(0,3,2,1).contiguous().view(n, p*c, s)).view(n, p, c, s)
           x = x.permute(1,0,2,3).contiguous()
           return x    


class st_grm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # [temporal kernel, sptial kernel]
                 stride = 1,
                 num_nodes=16,
                 gcn_dropout=0.
                 ):
        super().__init__()

        self.num_node = num_nodes
        self.nodevec1 =nn.Parameter(torch.randn(num_nodes, 64), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(64, num_nodes), requires_grad=True)
        
        self.dy1 = st_gcn_SpGraph(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                dropout=gcn_dropout)
        self.dy2 = st_gcn_SpGraph(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                dropout=gcn_dropout)
        self.st1 = static_module(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                num_nodes=num_nodes)
        self.st2 = static_module(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                num_nodes=num_nodes)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def adp_graph(self, max_num_neigh): # n c s p
        adp_A = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        threshold = 1/self.num_node
        tmp,_ = torch.kthvalue(-1 * adp_A, max_num_neigh + 1, dim=1, keepdim=True)
        bin_mask = (torch.logical_and((adp_A > threshold), (adp_A > -tmp)).type_as(adp_A) - adp_A).detach() + adp_A
        adp_A = adp_A*bin_mask
        return adp_A
    
    def forward(self, x): #(n, c, s, p)
        A_adp = self.adp_graph(1)
        d, _ = self.dy1(x, A_adp)
        d, _ = self.dy2(d, A_adp)
        d = torch.max(d, 2)[0]
        x = x.permute(3, 0, 1, 2).contiguous() # (p n c s)
        s = self.st2(self.st1(x))
        s = torch.max(s.permute(1, 2, 3, 0).contiguous(), 2)[0]
        out = torch.cat([d, s], 2)
        return out