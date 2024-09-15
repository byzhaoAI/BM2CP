import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        x = torch.einsum('bmcxy, bnmxy -> bncxy', (x, A))
        return x.contiguous()


class Mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(Mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Sequential(
            nn.Conv2d((gdep+1)*c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(c_out),
            nn.GELU(),
        )
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self, x, a):
        h = x
        out = [h]
        # x = torch.Size([b, m, c, x, y])
        # a = torch.Size([b, m, m, x, y])
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=2)
        m, b, z, h, w = ho.shape
        ho = ho.view(-1, z, h, w)
        ho = self.mlp(ho)
        ho = ho.view(m, b, -1, h, w)
        return ho


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, freeze=False):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes

        self.proj_evc1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.proj_evc2 = nn.Conv2d(dim, dim, 3, 1, 1)

        self.alpha = alpha
        self.adj = 0

        if freeze:
            print('freeze adjacent matrix')
            self.fix_network()

    def fix_network(self):
        """
        Fix the parameters of network during finetune stage.
        """
        for p in self.emb1.parameters():
            p.requires_grad = False

        for p in self.emb2.parameters():
            p.requires_grad = False

        for p in self.lin1.parameters():
            p.requires_grad = False
        
        for p in self.lin2.parameters():
            p.requires_grad = False


    def forward(self, pts):
        # pts shape: b, m, z, x, y
        b, m, z, x, y = pts.shape
        pts = pts.view(b*m, z, x, y)
        nodevec1, nodevec2 = self.proj_evc1(pts), self.proj_evc2(pts)
        nodevec1 = nodevec1.view(b, m, z, x, y)
        nodevec2 = nodevec2.view(b, m, z, x, y)

        nodevec1 = torch.tanh(self.alpha * nodevec1)
        nodevec2 = torch.tanh(self.alpha * nodevec2)

        a = torch.einsum('bmlxy,bnlxy->bmnxy', nodevec1, nodevec2) - torch.einsum('bmlxy,bnlxy->bmnxy', nodevec2, nodevec1)
        adj = torch.sigmoid(self.alpha * a)

        adj = adj + torch.eye(adj.size(1)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(adj.device)
        adj = adj / adj.sum(2).view(b, -1, 1, x, y)
        self.adj = adj
        # adj shape: b, m, m, x, y
        return adj
