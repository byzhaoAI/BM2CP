from re import A
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from opencood.utils.camera_utils import bin_depths


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
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


class ImgCamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample, ddiscr, mode, train=False, use_gt_depth=False, depth_supervision=True):
        super(ImgCamEncode, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision # in the case of not use gt depth
        self.chain_channels = 256 # 512

        if train:
            self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        else:
            self.trunk = EfficientNet.from_name('efficientnet-b0')
        print("Number of parameter EfficientNet: %d" % (sum([param.nelement() for param in self.trunk.parameters()])))

        self.up1 = Up(320+112, self.chain_channels)  # 上采样模块，输入输出通道分别为320+112和512
        if downsample == 8:
            self.up2 = Up(self.chain_channels+40, self.chain_channels)
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(self.chain_channels, self.D, kernel_size=1, padding=0)  # 1x1卷积，变换维度
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
        if self.downsample == 8:
            x = self.up2(x, endpoints['reduction_3'])
        return x  # x: 24 x 512 x 8 x 22


    def forward(self, x, depth_map):
        """
        Returns:
            log_depth : [B*N, D, fH, fW], or None if not used latter
            depth_gt_indices : [B*N, fH, fW], or None if not used latter
            new_x : [B*N, C, D, fH, fW]
        """
        # intrinsic.shape = torch.Size([4, 1, 3, 3])
        # coords_3d.shape = torch.Size([45191, 3])
        # pts -> pixel coord + depth
        # coords_2d, depths, valid_mask = self._forward(coords_3d, intrinsic, extrinsic)
        _, _, oriH, oriW = x.shape
        x_img = x[:,:3:,:,:]    # origin x: (B*num(cav), C, H, W)
        features = self.get_eff_features(x_img)     # 4x downscale feature: (B*num(cav), set_channels(e.g.256), H/4, W/4)
        x_img = self.image_head(features) #  8x downscale feature: B*N x C x fH x fW(24 x 64 x 8 x 22). C is the channel for next stage (i.e. bev)

        # resize depth
        batch, _, h, w = features.shape
        assert oriH % h == 0
        assert oriW % w == 0
        scaleh, scalew = oriH // h, oriW // w

        max_value = torch.max(depth_map)
        depth_map[depth_map<0] = max_value + 1
        pool_layer = nn.MaxPool2d(kernel_size=(scaleh, scalew), stride=(scaleh, scalew))
        depth_map = -1 * pool_layer(-1 * depth_map)
        depth_map[depth_map>max_value] = 0

        # generate one-hot refered ground truth
        depth_mask = ((depth_map) > 0).long()
        depth_map = depth_map.to(torch.int64).flatten(2).squeeze(1)
        one_hot_depth_map = []
        for batch_map in depth_map:
            one_hot_depth_map.append(F.one_hot(batch_map, num_classes=self.D))
        one_hot_depth_map = torch.stack(one_hot_depth_map).reshape(batch, h, w, self.D).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        depth_score = self.depth_head(features)
        depth_pred = F.softmax(depth_score, dim=1) # 对深度维进行softmax，得到每个像素不同深度的概率

        weighted_depth = one_hot_depth_map * 0.95 + depth_pred * 0.05
        weighted_depth = torch.ceil(weighted_depth) - ((weighted_depth - torch.floor(weighted_depth)) < 0.5).long()
        final_depth = depth_mask * weighted_depth + (1-depth_mask) * depth_pred

        new_x = final_depth.unsqueeze(1) * x_img.unsqueeze(2) # new_x: 24 x 64 x 41 x 8 x 18
        return x_img, new_x
    
    def _forward(self, xyz, int_matrix, ext_matrix, boundH=320, boundW=480, depth_min=0.1):
        xyz_hom = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=-1)   # (..., 3) -> (..., 4)[xyz+1]
        
        ext_matrix = torch.inverse(ext_matrix)[:,:,:3,:4]
        # img_pts = (int_matrix @ ext_matrix @ xyz_hom.T).T
        img_pts = (int_matrix @ ext_matrix @ xyz_hom.T).permute(0, 1, 3, 2)

        depth = img_pts[:,:,:,2]
        uv = img_pts[:,:,:,:2] / depth[:,:,:,None]
        uv_int = torch.round(uv)

        valid_mask = ((depth > depth_min) &
                      (uv_int[:,:,:,0] >= 0) & (uv_int[:,:,:,0] < boundH) & 
                      (uv_int[:,:,:,1] >= 0) & (uv_int[:,:,:,1] < boundW))

        return uv_int, depth, valid_mask    # [..., 2], [..., 1]
