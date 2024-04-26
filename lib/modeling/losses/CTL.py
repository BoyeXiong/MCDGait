import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper

class CTL_Loss(BaseLoss):
    def __init__(self, T_s=1, T_t=1, loss_term_weight=1.0):
        super(CTL_Loss, self).__init__(loss_term_weight)
        self.T_s = T_s
        self.T_t = T_t

    def loss_kld(self, y_s, y_t):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = F.softmax(y_t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean')
        return loss

    @gather_and_scale_wrapper
    def forward(self, f_s, f_t):  
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        #G_s = torch.nn.functional.normalize(G_s)
        G_s = F.softmax(G_s, dim=1) 
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        # G_t = torch.nn.functional.normalize(G_t)
        G_t = F.softmax(G_t, dim=1) 

        # G_diff = G_t - G_s
        # loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        loss = (self.loss_kld(G_s / self.T_s, G_t / self.T_t) +
                self.loss_kld(G_t / self.T_s, G_s / self.T_t)) / 2
        self.info.update({'loss': loss.detach().clone()}) 

        return loss, self.info

    