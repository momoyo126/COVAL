"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy
import cv2
from collections.abc import Sequence, Mapping

from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")




def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
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
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


class PC2ImgConverter(object):

    def __init__(self, XY=[512, 512], xRange=[-50, 50], yRange=[-50, 50], zRange=[-3, 3]):

        self.xRange = xRange
        self.yRange = yRange
        self.zRange = zRange
        self.XY = XY

    def getIdx(self, coords):
        coords = coords - np.array([[self.xRange[0], self.yRange[0], self.zRange[0]]])
        coords_xy = coords[:, 0:2]
        proj_x = coords_xy[:, 0] / (self.xRange[1] - self.xRange[0])
        proj_y = coords_xy[:, 1] / (self.yRange[1] - self.yRange[0])
        proj_x = np.floor(proj_x * self.XY[0]).astype(int)
        proj_y = np.floor(proj_y * self.XY[1]).astype(int)
        return proj_x, proj_y

    def getPavementArea(self, data_dict):
        coords = data_dict['coord']
        labels = data_dict['segment']

        in_bound_x = np.logical_and(self.xRange[0] < coords[:, 0], coords[:, 0] < self.xRange[1])
        in_bound_y = np.logical_and(self.yRange[0] < coords[:, 1], coords[:, 1] < self.yRange[1])
        in_bound_z = np.logical_and(self.zRange[0] < coords[:, 2], coords[:, 2] < self.zRange[1])
        in_bound_idx = np.logical_and(in_bound_x, np.logical_and(in_bound_y, in_bound_z))

        coords = coords[in_bound_idx]
        labels = labels[in_bound_idx]
        proj_x, proj_y = self.getIdx(coords)
        pavement_mask = (labels == 8) | (labels == 9) | (labels == 10)
        pavement_x = proj_x[pavement_mask]
        pavement_y = proj_y[pavement_mask]

        pavement_image = np.zeros(shape=(self.XY[1], self.XY[0]), dtype=np.float32)
        pavement_image[pavement_y, pavement_x] = 1

        return pavement_image, in_bound_idx, proj_x, proj_y

    def getPavementArea_allclass(self, data_dict):
        coords = data_dict['coord']
        coords[:, 0] = coords[:, 0].clip(min=self.xRange[0] + 0.01, max=self.xRange[1] - 0.01)
        coords[:, 1] = coords[:, 1].clip(min=self.yRange[0] + 0.01, max=self.yRange[1] - 0.01)
        height = coords[:, 2]
        indices = np.arange(height.shape[0])
        order = np.argsort(height)[::-1]

        proj_x, proj_y = self.getIdx(coords)

        pavement_image = np.zeros(shape=(self.XY[1], self.XY[0]), dtype=np.float32)
        pavement_image[proj_y[order], proj_x[order]] = height[order]

        return pavement_image, proj_x, proj_y

    def getPavementArea_class(self, data_dict, label, center=None):
        coords = data_dict['coord']
        labels = data_dict['segment']

        if center is not None:
            coords = coords - center.reshape(-1, 3)

        in_bound_x = np.logical_and(self.xRange[0] < coords[:, 0], coords[:, 0] < self.xRange[1])
        in_bound_y = np.logical_and(self.yRange[0] < coords[:, 1], coords[:, 1] < self.yRange[1])
        in_bound_z = np.logical_and(self.zRange[0] < coords[:, 2], coords[:, 2] < self.zRange[1])
        in_bound_idx = np.logical_and(in_bound_x, np.logical_and(in_bound_y, in_bound_z))

        coords = coords[in_bound_idx]
        labels = labels[in_bound_idx]
        proj_x, proj_y = self.getIdx(coords)
        pavement_mask = labels == label
        pavement_x = proj_x[pavement_mask]
        pavement_y = proj_y[pavement_mask]

        pavement_image = np.zeros(shape=(self.XY[1], self.XY[0]), dtype=np.float32)
        pavement_image[pavement_y, pavement_x] = 1

        return pavement_image, in_bound_idx, proj_x, proj_y

    def getPavementObject(self, segment, pavement_image, in_bound_idx, proj_x, proj_y):
        other_area = (segment == 11)
        labels = np.zeros(shape=in_bound_idx.shape, dtype=np.int32)
        pavement_object = pavement_image[proj_y, proj_x]
        labels[in_bound_idx] = pavement_object
        labels[other_area] = 0
        return labels

    def CompleteArea(self, pavement_image):
        # 先将图像转换为uint8类型（0-255范围）
        pavement_image_uint8 = (pavement_image * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bev_closed = cv2.morphologyEx(pavement_image_uint8, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓并过滤小面积区域
        contours, _ = cv2.findContours(bev_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 5:
                cv2.drawContours(bev_closed, [cnt], -1, 0, thickness=-1)

        # 转换回浮点类型（0-1范围）
        bev_closed = bev_closed / 255.0
        return bev_closed

    def getPavementObjectComplete(self, segment, coords):
        pavement_image, in_bound_idx, proj_x, proj_y = self.getPavementArea({'coord': coords, 'segment': segment})
        pavement_image = self.CompleteArea(pavement_image)
        labels = self.getPavementObject(segment, pavement_image, in_bound_idx, proj_x, proj_y)
        return labels


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
