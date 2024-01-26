"""
This class is about swap fusion applications (also known as Fused Axial Attention)
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.bm2cp_v2_modules.base_transformer import FeedForward, PreNormResidual


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

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
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

    def forward(self, x, y):
        # x shape: b, h, w, w_h, w_w, c
        batch, height, width, window_height, window_width, channel = x.shape
        h = self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        y = rearrange(y, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q = self.to_q(x)
        k, v = self.to_kv(y).chunk(2, dim=-1)
        
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # add positional bias
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')
        # attention
        attn = self.softmax(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out
        # out = self.to_out(out)
        return rearrange(out, '(b x y) w1 w2 d -> b x y w1 w2 d', b=batch, x=height, y=width)


class GridBlock(nn.Module):
    def __init__(self, dim, dim_head, mlp_dim, window_size, drop_out):
        super(GridBlock, self).__init__()
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.grid1 = Grid_Attention(dim, dim_head, drop_out, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp1 = FeedForward(dim, mlp_dim, drop_out)
        
        self.norm3 = nn.LayerNorm(dim)
        self.grid2 = Grid_Attention(dim, dim_head, drop_out, window_size)
        self.norm4 = nn.LayerNorm(dim)
        self.mlp2 = FeedForward(dim, mlp_dim, drop_out)        

    def forward(self, x, y):
        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
        y1 = rearrange(y, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
        x = self.grid1(self.norm1(x), self.norm1(y1))
        x = self.mlp1(self.norm2(x))
        x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')

        x = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
        y1 = rearrange(y, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=self.window_size[1], w2=self.window_size[2])
        x = self.grid2(self.norm3(x), self.norm3(y1))
        x = self.mlp2(self.norm4(x))
        x = rearrange(x, 'b x y w1 w2 d -> b d (w1 x) (w2 y)')
        return x


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
        assert isinstance(window_size, list)
        assert len(window_size) == 3

        self.layers = nn.ModuleList([])
        for i in range(args['depth']):
            block = GridBlock(input_dim, dim_head, mlp_dim, window_size, drop_out)
            self.layers.append(block)
        """
        # mlp head
        self.mlp_head = nn.Sequential(
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )
        """

    def forward(self, x, y):
        for stage in self.layers:
            x = stage(x, y)
        return x
        # return self.mlp_head(x)
