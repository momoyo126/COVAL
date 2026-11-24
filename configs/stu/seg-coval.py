# COVAL training configuration restored from the original training log.
_base_ = ['../_base_/default_runtime.py']

weight = 'exp/stu/minknet34-syn/model/model_last.pth'
epoch = 3
eval_epoch = 3
param_dicts = [dict(keyword='classifier', lr=0.24), dict(keyword='', lr=0.024)]

model = dict(
    type='AnomSegmentor',
    backbone=dict(type='Minknet34-ts', nclasses=20, cr=1),
    criteria=[dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)],
    recon=dict(type='TinyFeatureReconNet', in_channels=832, latent_ratio=0.25),
    consis=None,
    mutil_scale_consis=True)
optimizer = dict(
    type='SGD', lr=0.24, weight_decay=0.0001, momentum=0.9, nesterov=True)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.24, 0.24, 0.024],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'STUDataset'
data_root = 'data/stu'
ignore_index = -1
names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
    'traffic-sign', 'outlier'
]
index_valid_keys = [
    'coord', 'color', 'normal', 'superpoint', 'strength', 'segment',
    'instance', 'occlusion_mask'
]
data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
        'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign', 'outlier'
    ],
    train=dict(
        type='STUDataset',
        split='train',
        data_root='data/stu',
        transform=[
            dict(
                type='Update',
                keys_dict=dict(index_valid_keys=[
                    'coord', 'color', 'normal', 'superpoint', 'strength',
                    'segment', 'instance', 'occlusion_mask'
                ])),
            dict(
                type='AnomalyGenerator',
                database_path=
                'data/anomaly'),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.95, 1.05]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.01, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'occlusion_mask',
                      'anomaly_mask_rv'),
                feat_keys=['coord', 'strength'])
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='STUDataset',
        split='val',
        data_root='data/stu',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=False),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=['coord', 'strength'])
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='STUDataset',
        split='val',
        data_root='data/stu',
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=False,
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=['coord', 'strength'])
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }]]),
        ignore_index=-1))
