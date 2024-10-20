from re import A
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from einops import rearrange


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        #x1 = self.up(x1)  # 对x1进行上采样
        #x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        #return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, C, downsample):
        super(CamEncode, self).__init__()
        self.C = C  # 64
        self.downsample = downsample
        self.chain_channels = 128 # 256 # 512

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        print("Number of parameter EfficientNet: %d" % (sum([param.nelement() for param in self.trunk.parameters()])))

        self.up1 = Up(320+112, self.chain_channels)  # 上采样模块，输入输出通道分别为320+112和512
        if downsample <= 8:
            self.up2 = Up(self.chain_channels+40, self.chain_channels)
        if downsample <= 4:
            self.up3 = Up(self.chain_channels+24, self.chain_channels)
        if downsample <= 2:
            self.up4 = Up(self.chain_channels+16, self.chain_channels)
        if downsample == 1:
            self.up5 = nn.ConvTranspose2d(self.chain_channels, self.chain_channels, 2, 2, 0)

        self.image_head = nn.Conv2d(self.chain_channels, self.C, kernel_size=1, padding=0)

    def get_eff_features(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        if self.downsample <= 8:
            x = self.up2(x, endpoints['reduction_3'])
        if self.downsample <= 4:
            x = self.up3(x, endpoints['reduction_2'])
        if self.downsample <= 2:
            x = self.up4(x, endpoints['reduction_1'])
        if self.downsample == 1:
            x = self.up5(x)

        return x  # x: 24 x 512 x 8 x 22

    def forward(self, x):
        """
        x: [B*L, N, C, H, W]
        Returns:
            new_x : [B*N, C, D, fH, fW]
        """
        Num, n, c, h, w = x.shape
        x = x.view(Num * n, c, h, w)

        x_img = x[:,:3:,:,:]    # origin x: (B*num(cav), C, H, W)
        features = self.get_eff_features(x_img)     # 4x downscale feature: (B*num(cav), set_channels(e.g.256), H/4, W/4)
        x_img = self.image_head(features) #  8x downscale feature: B*N x C x fH x fW(24 x 64 x 8 x 22). C is the channel for next stage (i.e. bev)

        x_img = rearrange(x_img, '(l n) c h w -> l n c h w', l=Num, n=n)
        
        # 检查张量是否全为零，并且没有 NaN 或 Inf
        # is_all_zeros = (x_img.ne(0).logical_or(torch.isnan(x_img)).logical_or(torch.isinf(x_img))).sum() == 0
        return x_img
