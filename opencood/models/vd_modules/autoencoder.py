import torch
import torch.nn as nn

# PyTorch中的所有神经网络模型都应该继承自nn.Module基类，并进行初始化。
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # 编码器部分，高维->低维。
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), # 第一层全连接层，将输入的784维数据（即28*28像素的图像展平成向量）压缩到128维。
            nn.ReLU(), # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Linear(128, 64), # 第二层全连接层，进一步将数据从128维压缩到64维。
            nn.ReLU(), # 再次使用ReLU激活函数。
            nn.Linear(64, 12), # 第三层全连接层，将数据从64维压缩到12维。
            nn.ReLU(), # 再次使用ReLU激活函数。
            nn.Linear(12, 3) # 最后一层全连接层，将数据最终压缩到3维，得到编码后的数据。
        )
        
        # 解码器部分，低维->高维。
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(), # 使用ReLU激活函数。
            nn.Linear(12, 64), # 第二层全连接层，将数据从12维扩展到64维。
            nn.ReLU(), # 再次使用ReLU激活函数。
            nn.Linear(64, 128), # 第三层全连接层，将数据从64维扩展到128维。
            nn.ReLU(), # 再次使用ReLU激活函数。
            nn.Linear(128, 28*28), # 最后一层全连接层，将数据从128维扩展回784维，即原始图像大小。
            # nn.Sigmoid() # 使用Sigmoid激活函数，将输出压缩到0到1之间，因为原始图像在输入之前经过了标准化。
        )

    def forward(self, x):
        x = self.encoder(x) # 将输入数据通过编码器压缩。
        x = self.decoder(x) # 然后通过解码器进行重构。
        return x # 返回重构的数据。