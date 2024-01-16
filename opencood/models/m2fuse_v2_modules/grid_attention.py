"""
This class is about swap fusion applications (also known as Fused Axial Attention)
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

from opencood.models.m2fuse_v2_modules.base_transformer import FeedForward, PreNormResidual


class Grid_Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    window_size: int
        The window size in z, y, x axis.
    """

    def __init__(self, dim, dim_head=32, dropout=0., window_size=[1,7,7]):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = window_size

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # self.to_out = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.Dropout(dropout))

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, z_size, height, width, window_height, window_width, channel, = x.shape
        h = self.heads

        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))
        
        # attention
        attn = self.softmax(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)', l=z_size, w1=window_height, w2=window_width)

        # combine heads out
        # out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d', b=batch, x=height, y=width)


class GridAttEncoder(nn.Module):
    """
    Data rearrange -> swap block -> mlp_head
    Attention Fusion Blocks which contain window attention and grid attention.
    """

    def __init__(self, args):
        super(GridAttEncoder, self).__init__()
        # block related
        input_dim = args['dim']
        dim_head = args['dim_head']
        mlp_dim = args['mlp_dim']

        drop_out = args['drop_out']
        window_size = args['window_size']
        self.window_size = window_size
        assert isinstance(window_size, list)
        assert len(window_size) == 3

        self.local_layers, self.grid_layers = nn.ModuleList([]), nn.ModuleList([])
        self.local_projs, self.grid_projs = nn.ModuleList([]), nn.ModuleList([])

        for i in range(args['depth']):
            self.local_layers.append(PreNormResidual(input_dim, Grid_Attention(input_dim, dim_head, drop_out, window_size)))
            self.local_projs.append(PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)))
            
            self.grid_layers.append(PreNormResidual(input_dim, Grid_Attention(input_dim, dim_head, drop_out, window_size)))
            self.grid_projs.append(PreNormResidual(input_dim, FeedForward(input_dim, mlp_dim, drop_out)))

    def forward(self, x, mask):
        for local_att, local_proj, grid_att, grid_proj in zip(self.local_layers, self.local_projs, self.grid_layers, self.grid_projs):
            
            _mask =  rearrange(mask, 'b l e (x w1) (y w2) -> b x y w1 w2 e l', w1=self.window_size[1], w2=self.window_size[2])
            x = rearrange(x, 'b d m (x w1) (y w2) -> b m x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
            x = local_att(x, mask=_mask)
            x = local_proj(x)
            x = rearrange(x, 'b m x y w1 w2 d -> b d m (x w1) (y w2)')

            _mask = rearrange(mask, 'b l e (w1 x) (w2 y) -> b x y w1 w2 e l', w1=self.window_size[1], w2=self.window_size[2])
            x = rearrange(x, 'b d m (w1 x) (w2 y) -> b m x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
            x = grid_att(x, mask=_mask)
            x = grid_proj(x)
            x = rearrange(x, 'b m x y w1 w2 d -> b d m (w1 x) (w2 y)')

        return x
