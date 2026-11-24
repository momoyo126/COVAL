"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import yaml

from pathlib import Path
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class STUDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, sweep=1, sequence_root="data/STU/validation", **kwargs):
        self.ignore_index = ignore_index
        self.sweep = sweep
        self.sequence_root = Path(sequence_root)
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def _preload_poses(self):
        """Load and cache all pose transformations for each frame in each scene."""
        poses = list()
        for scene in self.scenes:
            scene_path = Path(scene[0]).parent.parent
            calib_file = scene_path / "calib.txt"
            pose_file = scene_path / "poses.txt"
            calibration = self.parse_calibration(calib_file)
            # Load poses for each frame in the scene
            poses.append(self.parse_poses(pose_file, calibration))
        return poses

    def _load_scenes(self):
        """Load all scenes and their respective frames based on mode."""
        scene_data = []
        mode = self.split
        if mode == "val":
            scene_ids = [8]
            # scene_ids = [32,201]
        elif mode == "test":
            scene_ids = [125, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 169]

        print(f"{mode}: {scene_ids}, {len(scene_ids)}")
        for scene_id in scene_ids:
            scene_path = str(self.sequence_root / f"{int(scene_id):02}")
            print(Path(scene_path + '/' + "velodyne"))
            frames = list(Path(scene_path + '/' + "velodyne").glob("*.bin"))
            frames = sorted(frames)
            scene_data.append([str(frame) for frame in frames])
        return scene_data

    def get_data_list(self):
        if 'train' == self.split:
            mode = 'train'
            self.data = self._load_yaml(self.data_root + '/' + f"{mode}_database.yaml")
            data = [[]]
            last_scene = self.data[0]["scene"]
            for x in self.data:
                if x["scene"] == last_scene:
                    data[-1].append(x)
                else:
                    last_scene = x["scene"]
                    data.append([x])
            for i in range(len(data)):
                data[i] = list(self.chunks(data[i], self.sweep))
            self.data = [val for sublist in data for val in sublist]
        else:
            self.scenes = self._load_scenes()
            self.pose_cache = self._preload_poses()
            self.data = []
            for scene_idx, scene in enumerate(self.scenes):
                for frame_idx in range(len(scene)):
                    frame_path = Path(scene[frame_idx])
                    label_path = frame_path.parent.parent / "labels" / (frame_path.stem + ".label")
                    self.data.append([{
                        "filepath": str(frame_path),
                        "label_filepath": str(label_path),
                        "pose": self.pose_cache[scene_idx][frame_idx].tolist(),
                        "scene": frame_path.parent.parent.name
                    }])
        return self.data

    def _find_scene_frame(self, idx):
        """Determine which scene and frame corresponds to a dataset index."""
        cumulative = 0
        for scene_idx, frames in enumerate(self.scenes):
            if idx < cumulative + len(frames):
                return scene_idx, idx - cumulative
            cumulative += len(frames)
        return -1, -1

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def chunks(self, lst, n):
        if "train" in self.split or n == 1:
            for i in range(len(lst) - n + 1):
                yield lst[i: i + n]
        else:
            for i in range(0, len(lst) - n + 1, n - 1):
                yield lst[i: i + n]
            if i != len(lst) - n:
                yield lst[i + n - 1:]

    def get_data(self, idx: int):
        coordinates_list = []
        strength_list = []
        features_list = []
        labels_list = []
        acc_num_points = [0]

        for time, scan in enumerate(self.data[idx % len(self.data)]):
            data_path = scan["filepath"]
            with open(data_path, "rb") as b:
                points = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
            coordinates = points[:, :3]
            strength = points[:, -1].reshape([-1, 1])
            strength_list.append(strength)
            coordinates_list.append(coordinates)
            acc_num_points.append(acc_num_points[-1] + len(coordinates))
            features = points[:, 3:4]
            time_array = np.ones((features.shape[0], 1)) * time
            features = np.hstack((time_array, features))
            features_list.append(features)
            if "test" in self.split:
                labels = np.zeros_like(features).astype(np.int64)
                labels_list.append(labels)
            else:
                label_path = scan["label_filepath"]
                with open(label_path, "rb") as a:
                    panoptic_label = np.fromfile(a, dtype=np.uint32)
                    semantic_label = panoptic_label & 0xFFFF
                    semantic_label = np.vectorize(self.learning_map.__getitem__)(semantic_label)
                labels = np.hstack((semantic_label[:, None], panoptic_label[:, None]))
                labels_list.append(labels)
        coordinates = np.vstack(coordinates_list)
        strength = np.vstack(strength_list)
        features = np.vstack(features_list)
        labels = np.vstack(labels_list)

        data_dict = dict(
            coord=coordinates,
            strength=strength,
            segment=labels[:,0].copy(),
            panoptic=labels[:,1].copy(),
            labels=labels,
            features=features,
            num_points=acc_num_points,
            sequence=scan["scene"],
            name=self.get_data_name(idx),
        )

        return data_dict

    def get_data_name(self, idx):
        scan = self.data[idx % len(self.data)][-1]
        file_path = scan['filepath']
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def parse_calibration(filename):
        calib = {}
        with open(filename) as calib_file:
            for line in calib_file:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(values).reshape(3, 4)
                calib[key] = pose
        return calib

    @staticmethod
    def parse_poses(filename, calibration):
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)
        poses = list()
        with open(filename) as file:
            for line in file:
                values = [float(v) for v in line.strip().split()]
                pose = np.eye(4)
                pose[:3, :4] = np.array(values).reshape(3, 4)
                pose = Tr_inv @ pose @ Tr
                poses.append(pose)
        return poses

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            2: ignore_index,  # "anomaly" mapped to "unlabeled" ---------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
