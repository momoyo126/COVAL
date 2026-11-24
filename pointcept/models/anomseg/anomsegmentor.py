import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import swanlab
import copy

from pointcept.models.losses import build_criteria
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch, batch2offset



def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "batch",
            "normal",
            "strength",
            "segment",
            "pano_segment",
            "instance",
            "proj_y",
            "proj_x",
            "feat",
            "grid_coord",
            "occlusion_mask"
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@MODELS.register_module()
class AnomSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None, recon=None, consis=None, mutil_scale_consis=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.recon = recon
        self.consis = consis
        self.mutil_scale_consis = mutil_scale_consis
        if recon:
            self.recon_head = build_model(recon)
            self.H = 128
            self.W = 2048
            self.fov_up = 22.5
            self.fov_down = -22.5
            self.range_project = LaserScan(H=self.H, W=self.W, fov_up=self.fov_up, fov_down=self.fov_down)

    def forward(self, input_dict):
        self.swanlab_results = {}
        if self.recon and self.training:
            seg_logits, aux_loss = self.recon_occlusion(input_dict)
        else:
            seg_logits = self.backbone(input_dict)
            aux_loss = 0
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            self.swanlab_results['loss/loss_seg'] = loss
            if swanlab.get_run() is not None:
                swanlab.log(self.swanlab_results)
            return dict(loss=loss+aux_loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


    def recon_occlusion(self,input_dict):
        aux_loss = 0
        offset = input_dict["offset"]
        batch_size = len(offset)
        batch = offset2batch(offset)

        occlusion_mask = input_dict['occlusion_mask']
        anomaly_mask_rv = input_dict['anomaly_mask_rv']
        inlier_mask = input_dict['segment'] != 19
        inlier_mask_ = inlier_mask[occlusion_mask]
        occlusion_inlier_mask = occlusion_mask & inlier_mask

        input_dict['batch'] = batch
        input_dict_1 = index_operator(input_dict,occlusion_mask,duplicate=True)
        input_dict_1['offset'] = batch2offset(input_dict_1['batch'])
        input_dict_2 = input_dict
        seg_logits_1, layers_to_fuse, feat_mutil_1 = self.backbone.forward_anom(input_dict_1)

        with torch.no_grad():
            layers_to_fuse_2, feat_mutil_2 = self.backbone.forward_anom_without_logits(input_dict_2)
            y4F = layers_to_fuse_2[0].F

        rv1 = self.proj(input_dict_1['batch'][inlier_mask_], input_dict_1['feat'][inlier_mask_,:3] + 0.001, feat_mutil_1[inlier_mask_],
                        batch_size)
        rv2 = self.proj(batch[inlier_mask], input_dict_2['feat'][:,:3][inlier_mask] + 0.001, feat_mutil_2[inlier_mask], batch_size)
        rv1_recon, _ = self.recon_head(rv1)
        anomaly_mask_rv = anomaly_mask_rv.repeat(1, rv1.shape[1], 1, 1)
        recon_loss = F.mse_loss(rv1_recon[anomaly_mask_rv], rv2[anomaly_mask_rv])
        aux_loss += recon_loss
        self.swanlab_results['loss/loss_recon'] = recon_loss
        if self.consis:
            consis_loss = self.consis_loss(feat_mutil_1[:,:96*2], feat_mutil_2[occlusion_mask][:,:96*2])
            aux_loss += consis_loss
        if self.mutil_scale_consis:
            mutil_scale_consis_loss = self.mutil_scale_consis_loss(layers_to_fuse_2, layers_to_fuse)
            aux_loss += mutil_scale_consis_loss
        input_dict['segment'] = input_dict['segment'][occlusion_mask]
        return seg_logits_1,aux_loss

    def proj(self, batch_idx, coordinates, feat1, b):
        rv1 = torch.zeros([b, self.H, self.W, feat1.shape[1]], dtype=feat1.dtype, device=feat1.device)
        with torch.no_grad():
            proj_x, proj_y, order = self.range_project.do_range_projection(coordinates)
        batch_idx = batch_idx[order]
        feat1 = feat1[order]
        rv1[batch_idx, proj_y, proj_x] = feat1
        rv1 = rv1.permute(0, 3, 1, 2)
        return rv1

    def consis_loss(self, feat1,feat2):
        consis_loss = F.mse_loss(feat1, feat2)
        self.swanlab_results['loss/loss_consis'] = consis_loss
        return consis_loss


    def mutil_scale_consis_loss(self, full_outputs, dropped_outputs):
        mutil_scale_consis_loss = self.backbone.compute_multiscale_consistency_loss(full_outputs, dropped_outputs)
        self.swanlab_results['loss/loss_mutil_scale_consis'] = mutil_scale_consis_loss
        return mutil_scale_consis_loss



class LaserScan:
    def __init__(self, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down

    def do_range_projection(self, points):
        """
        Project a pointcloud into a spherical projection image using PyTorch.
        
        Args:
            points: (N, 3) torch.Tensor, usually on GPU
        Returns:
            proj_x: (N,) torch.Tensor (int64)
            proj_y: (N,) torch.Tensor (int64)
            order: (N,) torch.Tensor (int64) - indices to sort by depth descending
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points)
            
        device = points.device
        fov_up = self.proj_fov_up / 180.0 * torch.pi 
        fov_down = self.proj_fov_down / 180.0 * torch.pi
        fov = abs(fov_down) + abs(fov_up)

        # 1. get depth of all points
        # numpy: np.linalg.norm(points, 2, axis=1)
        depth = torch.norm(points, p=2, dim=1)

        # 2. get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # 3. get angles of all points
        # numpy: -np.arctan2(scan_y, scan_x)
        yaw = -torch.atan2(scan_y, scan_x)
        
        # numpy: np.arcsin(scan_z / depth)
        pitch = torch.asin(torch.clamp(scan_z / (depth + 1e-8), -1.0, 1.0))

        # 4. get projections in image coords
        proj_x = 0.5 * (yaw / torch.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # 5. scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # 6. round and clamp for use as index
        # numpy: np.floor -> torch.floor
        proj_x = torch.floor(proj_x)
        proj_x = torch.clamp(proj_x, min=0, max=self.proj_W - 1).long() # .long() 转为 int64 用于索引
        
        self.proj_x = proj_x.clone()

        proj_y = torch.floor(proj_y)
        proj_y = torch.clamp(proj_y, min=0, max=self.proj_H - 1).long()
        self.proj_y = proj_y.clone()

        order = torch.argsort(depth, descending=True)
        
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        return proj_x, proj_y, order
