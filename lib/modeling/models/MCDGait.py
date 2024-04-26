import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..base_model import BaseModel
from ..modules import BasicConv2d, SetBlockWrapper
from ..gcn import Graph, Spatial_Basic_Block
from ..fusion_module import fusion_part_module
from ..st_grm import st_grm

class MCDGait(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):

        self.hidden_dim = model_cfg['hidden_dim']
        self.tta = model_cfg['tta']
        self.part_img = model_cfg['part_img']
        self.part_ske = model_cfg['part_ske']
        class_num = model_cfg['class_num']
        
        graph = Graph("coco")
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        ske_in_c = model_cfg['ske_channals']
        spatial_kernel_size = A.size(0)
        self.data_bn = nn.BatchNorm1d(ske_in_c[0] * A.size(1))

        
        self.gcn_networks = nn.Sequential(Spatial_Basic_Block(ske_in_c[0], ske_in_c[1],
                                                              spatial_kernel_size, False),
                                          Spatial_Basic_Block(ske_in_c[1], ske_in_c[1],
                                                              spatial_kernel_size),
                                          Spatial_Basic_Block(ske_in_c[1], ske_in_c[2],
                                                              spatial_kernel_size),
                                          Spatial_Basic_Block(ske_in_c[2], ske_in_c[2],
                                                              spatial_kernel_size),
                                          Spatial_Basic_Block(ske_in_c[2], ske_in_c[3],
                                                              spatial_kernel_size),
                                          Spatial_Basic_Block(ske_in_c[3], ske_in_c[3],
                                                              spatial_kernel_size))
        # initialize parameters for edge importance weighting
        edge_importance_weighting = model_cfg['edge_importance_weighting']
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
                for _ in self.gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_networks)

        img_in_c = model_cfg['img_channals']
        
        self.set_block1 = nn.Sequential(BasicConv2d(img_in_c[0], img_in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(img_in_c[1], img_in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(img_in_c[1], img_in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(img_in_c[2], img_in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(img_in_c[2], img_in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(img_in_c[3], img_in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_img*2, img_in_c[3], self.hidden_dim)))  #
        self.fc_bin1 = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(self.part_ske*2, ske_in_c[3], self.hidden_dim)))  
        

        # GCN
        self.st_sil = st_grm(in_channels=img_in_c[3],
                                 out_channels=img_in_c[3],
                                 kernel_size=[3, 1],
                                 num_nodes=self.part_img,
                                 )
        
        self.st_ske = st_grm(in_channels=ske_in_c[3],
                                 out_channels=ske_in_c[3],
                                 kernel_size=[3, 1],
                                 num_nodes=self.part_ske,
                                 )
        
        self.fusion_module = fusion_part_module(self.hidden_dim, class_num, self.part_img + self.part_ske)

    def hp(self, f, dim = 0):
        feature = f.mean(dim) + f.max(dim)[0]
        return feature

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0][0]  # [n, s, h, w]
        poses = ipts[1][0] # [n, s, c, v]

        N, T, H, W = sils.size()

        if not self.training and self.tta:
            sils_aug = torch.flip(sils,dims=[1])
            sils = torch.cat([sils, sils_aug], dim = 0)
            poses_aug = torch.flip(poses,dims=[1])
            poses = torch.cat([poses, poses_aug], dim = 0)

        sils = sils.unsqueeze(2)
        del ipts

        outs1 = self.set_block1(sils)
        outs2 = self.set_block2(outs1)
        outs3 = self.set_block3(outs2) # n, s, c, h, w
        x_1_s = self.hp(outs3.permute(4, 0, 2, 1, 3), dim=0)  # (p, n, c, s)
                                                              # (n, c, s, p)   1, 2, 3, 0
        x_1 = self.st_sil(x_1_s)
        x_1 = x_1.permute(2, 0, 1).contiguous()

        n, s, v, c = poses.size()
        poses = poses.permute(0, 2, 3, 1).contiguous()
        poses = poses.view(n, v * c, s)

        poses = self.data_bn(poses)
        poses = poses.view(n, v, c, s)

        poses = poses.permute(0, 2, 3, 1).contiguous()
        poses = poses.view(n, c, s, v)

        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            poses, _ = gcn(poses, self.A * importance)    #(n, c, s, k) k:v

        y_1 = self.st_ske(poses)
        y_1 = y_1.permute(2, 0, 1).contiguous()
        
        x_1 = x_1.matmul(self.fc_bin)  # 16 64 256
        y_1 = y_1.matmul(self.fc_bin1) # 17 64 256 

        embed_1 = self.fusion_module(x_1, y_1) # 

        if not self.training and self.tta:
            f1, f2 = torch.split(embed_1, [N, N], dim = 0) # n c p
            embed_1 = torch.mean(torch.stack([f1, f2]), dim=0)

        embed_1 = embed_1.permute(0, 2, 1).contiguous() # n p c
        sil_embed = x_1.permute(1, 0, 2).contiguous() # n p c
        ske_embed = y_1.permute(1, 0, 2).contiguous()

        n, s, c, h, w = sils.size()

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'triplet_sil': {'embeddings': sil_embed, 'labels': labs},
                'triplet_ske': {'embeddings': ske_embed, 'labels': labs},
                'ctl_sil': {'f_s': sil_embed, 'f_t': embed_1},
                'ctl_ske': {'f_s': ske_embed, 'f_t': embed_1},
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval