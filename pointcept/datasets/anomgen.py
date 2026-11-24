import numpy as np
import cv2
import yaml
from pathlib import Path
from scipy.spatial import KDTree
from random import random, choice, uniform
from .transform import PC2ImgConverter, ElasticDistortion, RandomScale, RandomRotate, TRANSFORMS
# from .laserscan import LaserScan

EDTrans = ElasticDistortion()
RSTrans = RandomScale(scale=[0.05, 0.3], anisotropic=True)
RRTrans = RandomRotate(angle=[-np.pi, np.pi])



@TRANSFORMS.register_module()
class AnomalyGenerator:
    def __init__(self, database_path='data/anomaly', mode='train', instance_database=None):
        self.XY = [200, 200]
        self.xRange = [-50, 50]
        self.yRange = [-50, 50]
        self.zRange = [-50, 50]
        self.bev_project = PC2ImgConverter(XY=self.XY, xRange=self.xRange, yRange=self.yRange, zRange=self.zRange)

        self.H = 128
        self.W = 2048
        self.fov_up = 22.5
        self.fov_down = -22.5
        self.range_project = LaserScan(H=self.H, W=self.W, fov_up=self.fov_up, fov_down=self.fov_down)

        self.range_unproj = RangeImageToPointCloud()

        self.database_path = Path(database_path)
        database_name = instance_database or f"{mode}_instances_database_pointcept.yaml"
        # Official preprocessing only creates the non-Pointcept index.
        if instance_database is None and not (self.database_path / database_name).is_file():
            database_name = f"{mode}_instances_database.yaml"
        self.instance_data = self._load_yaml(self.database_path / database_name)
        if not self.instance_data:
            raise ValueError("The anomaly instance database must not be empty")

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file

    def CompleteArea(self, pavement_image):
        # 先将图像转换为uint8类型（0-255范围）
        pavement_image_uint8 = (pavement_image * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bev_closed = cv2.morphologyEx(pavement_image_uint8, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓并过滤小面积区域
        contours, _ = cv2.findContours(bev_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 2:
                cv2.drawContours(bev_closed, [cnt], -1, 0, thickness=-1)

        # 转换回浮点类型（0-1范围）
        bev_closed = bev_closed / 255.0
        return bev_closed

    def generate_cube(self, center, size, resolution=0.01):
        """
        生成立方体点云
        center: 立方体中心点坐标 [x, y, z]
        size: 立方体边长
        resolution: 点云分辨率
        """
        x = np.arange(center[0] - size / 2, center[0] + size / 2, resolution)
        y = np.arange(center[1] - size / 2, center[1] + size / 2, resolution)
        z = np.arange(center[2] - size / 2, center[2] + size / 2, resolution)

        # 创建立方体表面点云
        points = []
        # 前面和后面
        for xi in x:
            for yi in y:
                points.append([xi, yi, center[2] + size / 2])
                points.append([xi, yi, center[2] - size / 2])
        # 左面和右面
        for yi in y:
            for zi in z:
                points.append([center[0] + size / 2, yi, zi])
                points.append([center[0] - size / 2, yi, zi])
        # 上面和下面
        for xi in x:
            for zi in z:
                points.append([xi, center[1] + size / 2, zi])
                points.append([xi, center[1] - size / 2, zi])

        coords = np.array(points)
        intensity = np.ones(len(points))
        return np.stack([coords, intensity], axis=1)

    def generate_sphere(self, center, radius, resolution=0.01):
        """
        生成球体点云
        center: 球体中心点坐标 [x, y, z]
        radius: 球体半径
        resolution: 点云分辨率
        """
        theta = np.arange(0, np.pi, resolution)
        phi = np.arange(0, 2 * np.pi, resolution)

        theta, phi = np.meshgrid(theta, phi)
        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)

        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        intensity = np.ones(len(points))
        return np.stack([points, intensity], axis=1)

    def select_random_instance(self):
        while 1:
            instance_dict = choice(self.instance_data)
            for _ in range(len(instance_dict["filepaths"])):
                idx = np.random.randint(len(instance_dict["filepaths"]))
                filepath = instance_dict["filepaths"][idx]
                instance = np.load(self.resolve_instance_path(filepath))
                if instance.shape[0] > 20:
                    break
            if instance.shape[0] > 20:
                break
        return instance

    def resolve_instance_path(self, filepath):
        path = Path(filepath)
        if path.is_file():
            return path
        candidate = self.database_path / path
        if candidate.is_file():
            return candidate
        # Archived databases may use data/instances or data/stu/instances.
        candidate = self.database_path / "instances" / path.name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Cannot locate instance {filepath!r} under {self.database_path}")

    def transform(self, anomaly_points, num_bins=20, threshold_ratio=0.1):
        """
        1. 对物体进行旋转变换
        2. 对物体进行缩放变换
        3. 对物体进行扭曲变换
        4. 对异常物体点云的z特征进行直方图划分，从低到高开始，只从比较大的区间开始取值

        参数:
            anomaly_points: 异常物体点云，形状为[N, 4]的numpy数组
            num_bins: 直方图的区间数量
            threshold_ratio: 判断区间大小的阈值比例（相对于总点数）

        返回:
            transformed_points: 处理后的点云
        """
        points = dict(coord=anomaly_points[:, :3])
        # 1. 旋转
        points = RRTrans(points)

        # 2. 扭曲
        points = EDTrans(points)

        # 3. 缩放
        points = RSTrans(points)

        anomaly_points[:, :3] = points['coord']

        # 4. 直方图过滤
        # 获取z坐标
        z_values = anomaly_points[:, 2]

        # 创建直方图
        hist, bin_edges = np.histogram(z_values, bins=num_bins)

        # 计算区间中点作为区间代表值
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 按z值从低到高排序区间索引
        sorted_indices = np.argsort(bin_centers)

        # 计算阈值（总点数乘以阈值比例）
        total_points = len(anomaly_points)
        min_points_threshold = total_points * threshold_ratio

        # 找出从低到高第一个比较大的区间的索引
        start_index = 0
        for idx in sorted_indices:
            if hist[idx] >= min_points_threshold:
                start_index = idx
                break

        # 获取起始区间对应的z值范围
        start_z = bin_edges[start_index]

        # 只保留z值大于等于起始z值的点
        mask = z_values >= start_z
        transformed_points = anomaly_points[mask]

        return transformed_points

    def detect_road_plane(self, point_cloud, label=8):
        """
        检测路面平面
        point_cloud: 点云数据字典，包含'coord'和'segment'
        """
        # BEV投影获得路面点云
        bev_mask, in_bound_idx, proj_x, proj_y = self.bev_project.getPavementArea_class(point_cloud, label=label)

        road = dict(
            mask=bev_mask,
            in_bound_idx=in_bound_idx,
            proj_x=proj_x,
            proj_y=proj_y,
        )

        return road

    def place_anomaly_on_road(self, point_cloud, anomaly_points, road, height_offset=0.01):
        """
        将异常物体放置在路面上
        point_cloud: 原始点云数据
        anomaly_points: 异常物体点云
        road: 包含路面bev信息的字典
        height_offset: 物体底部与路面的距离
        """
        o_center = point_cloud['coord'].mean(axis=0)
        anomaly_points[:, 2] = anomaly_points[:, 2] - 10 - o_center[2]
        anomaly_points_z = anomaly_points[:, 2].min()

        # 随机选择一个路面点
        valid_pts = np.argwhere(road['mask'])
        random_idx = np.random.randint(0, len(valid_pts))
        y, x = valid_pts[random_idx]

        # 获取该像素内所有的路面coord
        in_bound_idx = road['in_bound_idx']
        proj_x = road['proj_x']
        proj_y = road['proj_y']
        label_mask = (proj_x == x) & (proj_y == y)
        pavement_coord = point_cloud['coord'][in_bound_idx][label_mask]

        # 根据x, y计算该像素的范围, 并随机生成放置坐标
        x_resolution = (self.xRange[1] - self.xRange[0]) / self.XY[0]
        y_resolution = (self.yRange[1] - self.yRange[0]) / self.XY[1]
        x_range = np.array([self.xRange[0] + x * x_resolution, self.xRange[0] + (x + 1) * x_resolution])
        y_range = np.array([self.yRange[0] + y * y_resolution, self.yRange[0] + (y + 1) * y_resolution])
        center_x = np.random.uniform(x_range[0], x_range[1])
        center_y = np.random.uniform(y_range[0], y_range[1])

        # 计算与（center_x, center_y）最近的路面点的高度
        distances = np.linalg.norm(pavement_coord - np.array([center_x, center_y, anomaly_points_z]), axis=1)
        closest_idx = np.argmin(distances)
        closest_road_point = pavement_coord[closest_idx]
        closest_road_point_z = closest_road_point[2]

        # 异常物体的最低放置坐标为
        place_coord = np.array([center_x, center_y, closest_road_point_z + height_offset])

        # 放置异常物体，其最低坐标与place_coord相同
        anomaly_points[:, :3] -= anomaly_points[:, :3].min(axis=0)
        anomaly_points[:, :3] += place_coord

        return anomaly_points

    def apply_lidar_occlusion(self, point_cloud, anomaly_points, occlusion=True):
        """
        应用LIDAR遮挡效果
        point_cloud: 原始点云数据
        anomaly_points: 异常物体点云
        lidar_pos: LIDAR传感器位置
        """
        # 计算异常物体的range-view
        self.range_project.set_points(anomaly_points[:, :3] + 0.001, None, None)
        anomaly_mask = self.range_project.proj_mask.astype(bool)

        anomaly_mask_c = self.CompleteArea(anomaly_mask) > 0
        anomaly_mask = anomaly_mask | anomaly_mask_c

        # 计算原始点云的range-view
        self.range_project.set_points(point_cloud['coord'] + 0.001, None, None)
        original_mask = self.range_project.proj_mask.astype(bool)
        original_x, original_y = self.range_project.proj_x, self.range_project.proj_y

        mask = original_mask & ~anomaly_mask
        mask = mask[original_y, original_x]
        if occlusion:
            visible_original_points = index_operator(point_cloud, mask)
            mask = np.ones(visible_original_points['coord'].shape[0]).astype(bool)
        else:
            visible_original_points = point_cloud

        return visible_original_points, anomaly_points, mask, anomaly_mask

    def apply_lidar_occlusion_obj(self, anomaly_points):
        """
        应用LIDAR遮挡效果
        anomaly_points: 异常物体点云
        """
        # 计算异常物体的range-view
        self.range_project.set_points(anomaly_points[:, :3] + 0.001, None, None)

        anomaly_range_image = self.range_project.proj_range
        anomaly_intensity_image = self.range_project.proj_remission
        visible_anomaly_points = self.range_unproj.range_image_to_point_cloud_with_intensity(anomaly_range_image,
                                                                                             anomaly_intensity_image,
                                                                                             self.H, self.W,
                                                                                             self.fov_up, self.fov_down,
                                                                                             True)

        return visible_anomaly_points

    def merge_point_clouds(self, original_cloud, anomaly_points, anomaly_instance_count, occlusion_mask=None,
                           anomaly_mask=None, anomaly_label=19):
        """
        合并原始点云和异常物体点云
        original_cloud: 原始点云数据字典
        anomaly_points: 异常物体点云
        anomaly_label: 异常物体的语义标签
        """

        # 合并坐标
        merged_coord = np.vstack([original_cloud['coord'], anomaly_points[:, :3]])
        merged_strength = np.vstack([original_cloud['strength'], anomaly_points[:, 3:]])
        merged_segment = np.hstack([original_cloud['segment'],
                                    np.full(len(anomaly_points[:, :3]), anomaly_label)])
        if occlusion_mask is not None:
            anomaly_m = np.ones(anomaly_points.shape[0]).astype(bool)
            merged_occlusion_mask = np.hstack([occlusion_mask, anomaly_m])
        else:
            merged_occlusion_mask = np.ones_like(merged_segment).astype(bool)

        if anomaly_mask is not None:
            merged_anomaly_mask = anomaly_mask
        else:
            merged_anomaly_mask = np.zeros_like(merged_segment).astype(bool)

        # 创建合并后的点云字典
        merged_cloud = {
            'coord': merged_coord,
            'strength': merged_strength,
            'segment': merged_segment,
            'occlusion_mask': merged_occlusion_mask,
            'anomaly_mask_rv': merged_anomaly_mask[None, None, :, :]
        }

        return merged_cloud

    def create_anomaly_scene(self, original_cloud, max_instance_id, anomaly_type='random', size=0.5, position=[5, 0, 0],
                             synmix=False):
        """
        创建包含异常物体的场景
        data_idx: 原始数据索引
        anomaly_type: 异常物体类型 ('cube' 或 'sphere')
        size: 异常物体大小
        position: 异常物体位置 [x, y, z]（z坐标会自动调整到路面）
        synmix: 是否使用synmix
        """
        # 加载原始点云数据, 字典{coord，strength, segment}

        # 检测路面平面
        # road_num = (original_cloud['segment'] == 9).sum()
        # print(road_num,' - ', np.unique(original_cloud['segment']))
        plane_model = self.detect_road_plane(original_cloud, label=8)

        anomaly_points_list = []
        anomaly_instance_count = []
        count = 0
        for _ in range(np.random.randint(2, 10)):
            # 生成异常物体
            if anomaly_type == 'cube':
                anomaly_points = self.generate_cube(position, size)
            elif anomaly_type == 'sphere':
                anomaly_points = self.generate_sphere(position, size / 2)
            elif anomaly_type == 'random':
                anomaly_points = self.select_random_instance()
            else:
                raise ValueError("不支持的异常物体类型")

            if synmix:
                anomaly_points_2 = self.select_random_instance()
                mix_idx = np.random.randint(0, anomaly_points.shape[0])
                mix_idx_2 = np.random.randint(0, anomaly_points_2.shape[0])
                mix_coord = anomaly_points[mix_idx][:3]
                mix_coord_2 = anomaly_points_2[mix_idx_2][:3]
                offset = mix_coord - mix_coord_2
                anomaly_points_2[:, :3] = anomaly_points_2[:, :3] + offset[None, :]
                anomaly_points = np.concatenate([anomaly_points, anomaly_points_2])
            # 对异常物体点云进行变换处理
            anomaly_points = self.transform(anomaly_points, threshold_ratio=0.2)  # [M, 4]

            # 将异常物体放置在路面上
            anomaly_points = self.place_anomaly_on_road(original_cloud, anomaly_points, plane_model)  # [M, 4]
            anomaly_points = self.apply_lidar_occlusion_obj(anomaly_points)
            anomaly_points_list.append(anomaly_points)
            anomaly_instance_count.append(np.zeros(anomaly_points.shape[0], dtype=np.int32) + count)
            count += 1

        anomaly_points = np.vstack(anomaly_points_list)
        anomaly_instance_count = np.hstack(anomaly_instance_count) + max_instance_id + 1

        # 应用LIDAR遮挡
        original_cloud, anomaly_points, occlusion_mask, anomaly_mask = self.apply_lidar_occlusion(original_cloud,
                                                                                                  anomaly_points,
                                                                                                  occlusion=False)

        # 合并点云
        merged_cloud = self.merge_point_clouds(original_cloud, anomaly_points, anomaly_instance_count, occlusion_mask,
                                               anomaly_mask)

        return merged_cloud


    def __call__(self, data_dict):
        index_valid_keys = data_dict['index_valid_keys']
        data_dict = self.create_anomaly_scene(data_dict, max_instance_id=100000)
        data_dict['index_valid_keys'] = index_valid_keys
        return data_dict


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "strength",
            "segment",
            "pano_segment",
            "instance",
            "proj_y",
            "proj_x"
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


@TRANSFORMS.register_module()
class SynAnomalyGenerator(AnomalyGenerator):
    """Baseline's archived generator: original metadata name and unmodified __call__."""

    def __init__(self, database_path='data/anomaly', mode='train'):
        super().__init__(database_path, mode, instance_database=f"{mode}_instances_database.yaml")

    def __call__(self, data_dict):
        return self.create_anomaly_scene(data_dict, max_instance_id=100000)


class LaserScan():
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0, if_drop=False, if_flip=False, if_scale=False, if_rotate=False,if_jitter=False):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()
        self.if_drop = if_drop
        self.if_flip = if_flip
        self.if_scale = if_scale
        self.if_rotate = if_rotate
        self.if_jitter = if_jitter

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z
        self.remissions = np.zeros((0, 1), dtype=np.float32)  # [m ,1]: remission
        self.segment = np.zeros((0,), dtype=np.float32)

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)
        self.proj_segment = np.full((self.proj_H, self.proj_W), -1,
                                    dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.int32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission


        self.set_points(points, remissions)

    def set_points(self, points, remissions=None, segment=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()
        # print(points.shape)
        # data aug (drop)
        if self.if_drop:
            max_num_drop = int(len(points) * 0.3)  # drop ~10%
            num_drop = np.random.randint(low=0, high=max_num_drop)
            self.points_to_drop = np.random.randint(low=0, high=len(points) - 1, size=max_num_drop)
            self.points_to_drop = np.unique(self.points_to_drop)
            points = np.delete(points, self.points_to_drop, axis=0)
            remissions = np.delete(remissions, self.points_to_drop)
            # print(max_num_drop)
        # print(points.shape)

        # data aug (flip)
        if self.if_flip:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]  # flip x
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]  # flip y
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]  # flip both x and y

        # data aug (scale)
        if self.if_scale:
            scale = 1.05  # [-5%, +5%]
            rand_scale = np.random.uniform(1, scale)
            if np.random.random() < 0.5:
                rand_scale = 1 / scale
            points[:, 0] *= rand_scale
            points[:, 1] *= rand_scale

        # data aug (rotate)
        if self.if_rotate:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # data aug (jitter)
        if self.if_jitter:
            jitter = 0.1
            rand_jitter = np.clip(np.random.normal(0, jitter, 3), -3 * jitter, 3 * jitter)
            points += rand_jitter
        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points  # get xyz
        self.segment = segment if segment is not None else -np.ones((points.shape[0]), dtype=np.int32)
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)
        # depth_mask = depth>1.6
        # depth = depth[depth_mask]
        # self.points = self.points[depth_mask]

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order
        unique_beams = np.unique(proj_y)
        # print(unique_beams, len(unique_beams))

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        # order = np.argsort(depth)
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        segment = self.segment[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_segment[proj_y, proj_x] = segment
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.int32)
        # self.proj_x = proj_x
        # self.proj_y = proj_y
        self.depth = depth
        self.order = order


class RangeImageToPointCloud:
    def __init__(self):
        pass

    def range_image_to_point_cloud(self, proj_range, H, W, fov_up, fov_down, add_noise=False, depth_noise_std=0.005,
                                   point_jitter_std=0.01):
        """
        从投影范围图像恢复点云

        参数:
            proj_range: 投影范围图像，形状为[H, W]，值为深度，-1表示无效点
            H: 图像高度
            W: 图像宽度
            fov_up: 上视角（度数）
            fov_down: 下视角（度数）
            add_noise: 是否添加噪声以模拟真实情况
            depth_noise_std: 深度噪声的标准差（米）
            point_jitter_std: 点云抖动的标准差（米）

        返回:
            points: 恢复的点云，形状为[N, 3]，包含有效点的3D坐标
        """
        # 转换角度为弧度
        fov_up_rad = fov_up / 180.0 * np.pi
        fov_down_rad = fov_down / 180.0 * np.pi
        fov_total = abs(fov_down_rad) + abs(fov_up_rad)

        # 创建网格坐标
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        if add_noise:
            noise = np.random.uniform(0, 1)
            x_noise = 0.8 * noise + 0.2 * np.random.uniform(0, 1, size=x_coords.shape)
            y_noise = 0.8 * noise + 0.2 * np.random.uniform(0, 1, size=y_coords.shape)
            y_coords = y_coords + y_noise
            x_coords = x_coords + x_noise

        # 计算每个像素对应的角度
        # 计算pitch角（垂直角度）
        proj_y_normalized = y_coords / H  # 转换到[0, 1]范围
        pitch = -(proj_y_normalized - 1) * fov_total - abs(fov_down_rad)

        # 计算yaw角（水平角度）
        proj_x_normalized = x_coords / W  # 转换到[0, 1]范围
        yaw = -(proj_x_normalized * 2 * np.pi - np.pi)  # 转换到[-pi, pi]范围

        # 筛选有效点（深度不为-1的点）
        valid_mask = proj_range != -1

        # 获取有效点的深度和角度
        valid_depth = proj_range[valid_mask]
        valid_pitch = pitch[valid_mask]
        valid_yaw = yaw[valid_mask]

        # 添加微小扰动以模拟真实情况
        if add_noise:
            # 为深度添加高斯噪声
            depth_noise = np.random.normal(0, depth_noise_std, size=valid_depth.shape)
            valid_depth = valid_depth + depth_noise

        # 从深度和角度计算3D坐标
        # x = r * cos(pitch) * cos(yaw)
        # y = r * cos(pitch) * sin(yaw)
        # z = r * sin(pitch)
        x = valid_depth * np.cos(valid_pitch) * np.cos(valid_yaw)
        y = valid_depth * np.cos(valid_pitch) * np.sin(valid_yaw)
        z = valid_depth * np.sin(valid_pitch)

        # 添加点云抖动
        # if add_noise:
        #     x += np.random.normal(0, point_jitter_std, size=x.shape)
        #     y += np.random.normal(0, point_jitter_std, size=y.shape)
        #     z += np.random.normal(0, point_jitter_std, size=z.shape)

        # 组合成点云
        points = np.vstack([x, y, z]).T

        return points

    def range_image_to_point_cloud_with_intensity(self, proj_range, proj_remission, H, W, fov_up, fov_down,
                                                  add_noise=False, depth_noise_std=0.005, angle_noise_std=0.005,
                                                  point_jitter_std=0.01, intensity_noise_std=0.05):
        """
        从投影范围图像和强度图像恢复带强度信息的点云

        参数:
            proj_range: 投影范围图像，形状为[H, W]，值为深度，-1表示无效点
            proj_remission: 投影强度图像，形状为[H, W]
            H: 图像高度
            W: 图像宽度
            fov_up: 上视角（度数）
            fov_down: 下视角（度数）
            add_noise: 是否添加噪声以模拟真实情况
            depth_noise_std: 深度噪声的标准差（米）
            point_jitter_std: 点云抖动的标准差（米）
            intensity_noise_std: 强度噪声的标准差

        返回:
            point_cloud: 字典，包含'coord'和'strength'
        """
        # 获取点云坐标（带噪声）
        points = self.range_image_to_point_cloud(proj_range, H, W, fov_up, fov_down, add_noise, depth_noise_std,
                                                 point_jitter_std)

        # 获取有效点的强度值
        valid_mask = proj_range != -1
        valid_remission = proj_remission[valid_mask].reshape(-1, 1)

        # 为强度添加噪声
        if add_noise:
            intensity_noise = np.random.normal(0, intensity_noise_std, size=valid_remission.shape)
            intensity_noise = 0.2 * np.random.normal(0, intensity_noise_std, size=[1, 1]) + 0.01 * intensity_noise
            valid_remission = valid_remission + intensity_noise
            # 确保强度值在合理范围内（0-1）
            valid_remission = np.clip(valid_remission, 0, 1)

        points = np.hstack((points, valid_remission))

        return points
