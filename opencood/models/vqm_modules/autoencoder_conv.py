import torch
import torch.nn as nn

# PyTorch中的所有神经网络模型都应该继承自nn.Module基类，并进行初始化。
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器部分，高维->低维。
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 第一层全连接层，将输入的784维数据（即28*28像素的图像展平成向量）压缩到128维。
            nn.ReLU(), # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 第一层全连接层，将输入的784维数据（即28*28像素的图像展平成向量）压缩到128维。
            nn.ReLU(), # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # 最后一层全连接层，将数据最终压缩到3维，得到编码后的数据。
            nn.ReLU(), # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # 最后一层全连接层，将数据最终压缩到3维，得到编码后的数据。
        )
        
        # 解码器部分，低维->高维。
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0), # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(), # 使用ReLU激活函数。
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0), # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(), # 使用ReLU激活函数。
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0), # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(), # 使用ReLU激活函数。
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0), # 最后一层全连接层，将数据从128维扩展回784维，即原始图像大小。
            # nn.Sigmoid() # 使用Sigmoid激活函数，将输出压缩到0到1之间，因为原始图像在输入之前经过了标准化。
        )

    def forward(self, x):
        med_x = self.encoder(x) # 将输入数据通过编码器压缩。
        x = self.decoder(med_x) # 然后通过解码器进行重构。
        return med_x, x # 返回重构的数据。