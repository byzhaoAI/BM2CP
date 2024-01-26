"""
Multi-scale window transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def get_relative_distances(window_size_x, window_size_y):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size_x) for y in range(window_size_y)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = dim_head * heads
        assert isinstance(window_size, list) and len(window_size) == 2

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size[0], window_size[1]) + torch.tensor(np.array(window_size)) - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] ** 2, window_size[1] ** 2))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        l, h, w, c, m = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        new_h = h // self.window_size[0]
        new_w = w // self.window_size[1]
        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        q, k, v = map(lambda t: rearrange(t, 'l (new_h w_h) (new_w w_w) (m c) -> l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size[0], w_w=self.window_size[1]), qkv)
        # b l m h window_size window_size
        dots = torch.einsum('l m h i c, l m h j c -> l m h i j', q, k) * self.scale
        # consider prior knowledge of the local window
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum('l m h i j, l m h j c -> l m h i c', attn, v)
        # b l h w c
        out = rearrange(out, 'l m (new_h new_w) (w_h w_w) c -> l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size[0], w_w=self.window_size[1], new_w=new_w, new_h=new_h)
        out = self.to_out(out)
        
        return out


class PyramidWindowAttention(nn.Module):
    def __init__(self, dim, window_size, args):
        super().__init__()
        # window_size = args['window_size']
        heads = args['heads']
        # dim_heads = args['dim_heads']
        dim_heads = [dim//head for head in heads]
        
        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim, head, dim_head, args['dropout'], ws, args['relative_pos_embedding']))
        
        if 'fusion_method' in args:
            self.fuse_method = args['fusion_method']
            if self.fuse_method == 'split_attn':
                self.split_attn = SplitAttn(dim)
        else:
            self.fuse_method = 'naive'

    def forward(self, x):
        # x: torch.Size([2, 64, 100, 252])
        x = rearrange(x, 'l c h w -> l h w c')
        if self.fuse_method == 'split_attn':
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            out = self.split_attn(window_list)
            return rearrange(out, 'l h w c -> l c h w')
        
        # naive fusion will just sum up all window attention output and do a mean
        output = None
        for wmsa in self.pwmsa:
            output = wmsa(x) if output is None else output + wmsa(x)
        return rearrange(output, 'l h w c -> l c h w') / len(self.pwmsa)       


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        # x: (L, 1, 1, 3C)
        cav_num = x.size(0)

        if self.radix > 1:
            # x: (L, 1, 3, C)
            x = x.view(cav_num, self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=2)
            # 3LC
            x = x.reshape(-1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    def __init__(self, input_dim):
        super(SplitAttn, self).__init__()
        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.bn1 = nn.LayerNorm(input_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, input_dim * 3, bias=False)

        self.rsoftmax = RadixSoftmax(3, 1)

    def forward(self, window_list):
        # window list: [(L, H, W, C) * 3]
        assert len(window_list) == 3, 'only 3 windows are supported'

        sw, mw, bw = window_list[0], window_list[1], window_list[2]
        L = sw.shape[0]

        # global average pooling, L, H, W, C
        x_gap = sw + mw + bw
        # L, 1, 1, C
        x_gap = x_gap.mean((1, 2), keepdim=True)
        x_gap = self.act1(self.bn1(self.fc1(x_gap)))
        # L, 1, 1, 3C
        x_attn = self.fc2(x_gap)
        # L, 1, 1, 3C
        x_attn = self.rsoftmax(x_attn).view(L, 1, 1, -1)

        out = sw * x_attn[:, :, :, 0:self.input_dim] + mw * x_attn[:, :, :, self.input_dim:2*self.input_dim] + bw * x_attn[:, :, :, self.input_dim*2:]

        return out
