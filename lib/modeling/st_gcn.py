import torch
import torch.nn as nn
import torch.nn.functional as F
                                                                                                
from .gcn import Diffusion_GCN                                                                                          

# SpatialGraphConv
class st_gcn_SpGraph(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,   # [temporal kernel, sptial kernel]
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        # assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # self.gcn = SpatialGraphConv(in_channels, out_channels, kernel_size=kernel_size[1])
        self.gcn = Diffusion_GCN(c_in=in_channels, c_out=out_channels, dropout=0.1)
        # self.gcn = GraphConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size[1], dropout=0.1)
        self.tcn = nn.Sequential(
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x = x.permute(0, 1, 3, 2).contiguous()
        x, A = self.gcn(x, A)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.tcn(x) + res

        return self.relu(x), A

