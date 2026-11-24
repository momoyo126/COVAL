import torch.nn as nn
from pointcept.models.builder import MODELS, build_model

@MODELS.register_module()
class TinyFeatureReconNet(nn.Module):
    """
    小型特征图重建网络
    输入: (B, C, H, W) 特征图
    输出: (B, C, H, W) 重建特征图
    """

    def __init__(self, in_channels, latent_ratio=0.5):
        super().__init__()

        # 潜在空间通道数
        latent_channels = max(1, int(in_channels * latent_ratio))

        self.linera1 = nn.Conv2d(in_channels, latent_channels, kernel_size=1, stride=1)
        self.linera2 = nn.Conv2d(latent_channels, in_channels, kernel_size=1, stride=1)

        # 编码器：空间下采样2倍 + 通道压缩
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels // 2, latent_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # 解码器：空间上采样2倍 + 通道扩展
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels // 2, latent_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels // 2, latent_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.linera1(x)
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        reconstructed = self.linera2(reconstructed)
        return reconstructed, latent