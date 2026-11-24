import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spF

from torchsparse import SparseTensor
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset
from .minknet import BasicConvolutionBlock, ResidualBlock, BasicDeconvolutionBlock, pad_to_multiple_of_4



@MODELS.register_module("Minknet34-ts")
class Minknet34(nn.Module):

    def __init__(self, nclasses=19, cr=0.5, in_channels=4):
        super().__init__()

        self.nclasses = nclasses
        self.in_channels = in_channels

        # minknet/voxel
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(self.in_channels, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], self.nclasses))

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        coord = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()

        # voxel encoder
        feat = pad_to_multiple_of_4(feat)
        if self.training:
            feat = feat.half()
        x = SparseTensor(feat, coord)
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        return out

    def forward_anom(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        coord = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()

        # voxel encoder
        feat = pad_to_multiple_of_4(feat)
        if self.training:
            feat = feat.half()
        x = SparseTensor(feat, coord)
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        # --- Multi-Scale Fusion (包含 x4) ---
        target_tensor = y4  # 目标坐标系 (1x)
        # 这里包含了: [1x, 2x, 4x, 8x, 16x]
        layers_to_fuse = [y4, y3, y2, y1, x4]

        fused_feats_list = []

        for layer in layers_to_fuse:
            if layer is target_tensor:
                fused_feats_list.append(layer.F)
            else:
                proj_feat = self.project_to_target(source_tensor=layer, target_tensor=target_tensor)
                fused_feats_list.append(proj_feat)

        # 拼接所有特征
        # 最终维度 = C_y4 + C_y3 + C_y2 + C_y1 + C_x4
        cat_feats = torch.cat(fused_feats_list, dim=1)

        out = self.classifier(y4.F)

        return out, layers_to_fuse, cat_feats

    def forward_anom_without_logits(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        batch = offset2batch(offset)

        coord = torch.cat(
            [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
        ).contiguous()

        # voxel encoder
        # feat[:,:3] = 0
        feat = pad_to_multiple_of_4(feat)
        if self.training:
            feat = feat.half()
        x = SparseTensor(feat, coord)
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        # --- Multi-Scale Fusion (包含 x4) ---
        target_tensor = y4  # 目标坐标系 (1x)
        layers_to_fuse = [y4, y3, y2, y1, x4]

        fused_feats_list = []

        for layer in layers_to_fuse:
            if layer is target_tensor:
                fused_feats_list.append(layer.F)
            else:
                proj_feat = self.project_to_target(source_tensor=layer, target_tensor=target_tensor)
                fused_feats_list.append(proj_feat)

        # 拼接所有特征
        # 最终维度 = C_y4 + C_y3 + C_y2 + C_y1 + C_x4
        cat_feats = torch.cat(fused_feats_list, dim=1)

        # out = self.classifier(y4.F)

        return layers_to_fuse, cat_feats

    @staticmethod
    def project_to_target(source_tensor: SparseTensor, target_tensor: SparseTensor):
        """
        使用 F.sphash 和 F.sphashquery 加速的多尺度特征投影

        Args:
            source_tensor: 低分辨率特征 (如 y1, y2...)
            target_tensor: 高分辨率目标特征 (如 y4)
        Returns:
            projected_feat: Tensor, shape [N_target, C_source]
        """
        source_coords = source_tensor.C
        target_coords = target_tensor.C
        scale = source_tensor.s[0] // target_tensor.s[0]

        if scale == 1:
            if source_tensor.F.shape[0] == target_tensor.F.shape[0]:
                return source_tensor.F

        target_coords_down = target_coords.clone()
        target_coords_down[:, 1:] //= scale

        source_hash = spF.sphash(source_coords)
        target_hash = spF.sphash(target_coords_down)
        idx = spF.sphashquery(target_hash, source_hash)
        output_feat = torch.zeros(
            (target_tensor.F.shape[0], source_tensor.F.shape[1]),
            dtype=source_tensor.F.dtype,
            device=source_tensor.F.device
        )
        mask = (idx != -1)

        if mask.any():
            output_feat[mask] = source_tensor.F[idx[mask]]

        return output_feat


    @staticmethod
    def align_feature_by_coords(full_tensor: SparseTensor, dropped_tensor: SparseTensor):
        """
        在 full_tensor 中寻找与 dropped_tensor 坐标相同的点，并提取其特征。
        用于对比学习或一致性 Loss 计算。

        Args:
            full_tensor: 完整点云经过 Backbone 某层的输出 (N_full, C)
            dropped_tensor: Drop 后点云经过 Backbone 同层的输出 (N_drop, C)

        Returns:
            aligned_full_feat: 从 full_tensor 中提取出的特征，形状与 dropped_tensor 相同 (N_drop, C)
        """
        full_coords = full_tensor.C
        drop_coords = dropped_tensor.C

        full_hash = spF.sphash(full_coords)
        drop_hash = spF.sphash(drop_coords)

        indices = spF.sphashquery(drop_hash, full_hash)

        mask = (indices != -1)

        aligned_full_feat = torch.zeros_like(dropped_tensor.F)

        if mask.all():
            aligned_full_feat = full_tensor.F[indices]
        else:
            valid_indices = indices[mask]
            aligned_full_feat[mask] = full_tensor.F[valid_indices]

        return aligned_full_feat

    def compute_multiscale_consistency_loss(self, full_outputs, dropped_outputs):
        """
        逐层计算一致性 Loss，节省显存。

        Args:
            full_outputs: list of SparseTensor [x1, x2, x3, x4...] from full input
            dropped_outputs: list of SparseTensor [x1, x2, x3, x4...] from dropped input
        """
        total_loss = 0.0
        num_layers = len(full_outputs)

        for i, (f_out, d_out) in enumerate(zip(full_outputs, dropped_outputs)):
            aligned_feat = self.align_feature_by_coords(f_out, d_out)
            layer_loss = F.mse_loss(aligned_feat, d_out.F)
            total_loss += layer_loss

        return total_loss / num_layers
