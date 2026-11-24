"""Build local STU metadata and instance banks without copying the raw dataset."""

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


THING_LABELS = {10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
                30: 6, 31: 7, 32: 8, 252: 1, 253: 7, 254: 6,
                255: 8, 256: 5, 257: 5, 258: 4, 259: 5}


def read_poses(scene):
    calibration = {}
    for line in (scene / 'calib.txt').read_text().splitlines():
        if not line.strip():
            continue
        key, values = line.split(':', 1)
        matrix = np.eye(4)
        matrix[:3] = np.fromstring(values, sep=' ').reshape(3, 4)
        calibration[key] = matrix
    transform = calibration['Tr']
    inverse = np.linalg.inv(transform)
    poses = []
    for line in (scene / 'poses.txt').read_text().splitlines():
        if not line.strip():
            continue
        pose = np.eye(4)
        pose[:3] = np.fromstring(line, sep=' ').reshape(3, 4)
        poses.append(inverse @ pose @ transform)
    return poses


def build(root, metadata, anomaly, extract_instances=True):
    root, metadata, anomaly = (Path(path).resolve() for path in (root, metadata, anomaly))
    scenes = sorted(path for path in (root / 'train').iterdir() if path.is_dir())
    if not scenes:
        raise FileNotFoundError(f'No training scenes under {root / "train"}')
    targets = [metadata / 'train_database.yaml']
    if extract_instances:
        targets += [anomaly / 'train_instances_database.yaml',
                    anomaly / 'train_instances_database_pointcept.yaml']
        if (anomaly / 'instances').exists():
            raise FileExistsError(f'Instance destination already exists: {anomaly / "instances"}')
    for path in targets:
        if path.exists():
            raise FileExistsError(f'Refusing to replace metadata: {path}')
    records, instances = [], OrderedDict()
    for scene in scenes:
        poses = read_poses(scene)
        scans = sorted((scene / 'velodyne').glob('*.bin'))
        if not scans:
            raise FileNotFoundError(f'No scans in {scene / "velodyne"}')
        for scan in tqdm(scans, desc=scene.name):
            label_path = scene / 'labels' / (scan.stem + '.label')
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            pose = poses[int(scan.stem)]
            records.append(dict(filepath=str(scan), label_filepath=str(label_path),
                                pose=pose.tolist(), scene=int(scene.name)))
            if not extract_instances:
                continue
            points = np.fromfile(scan, dtype=np.float32).reshape(-1, 4)
            labels = np.fromfile(label_path, dtype=np.uint32)
            if len(labels) != len(points):
                raise ValueError(f'Point/label count mismatch: {scan}')
            # Match the original instance preprocessing: save pose-aligned XYZI.
            points[:, :3] = points[:, :3] @ pose[:3, :3].T + pose[:3, 3]
            for panoptic in np.unique(labels):
                semantic, instance_id = int(panoptic) & 0xffff, int(panoptic) >> 16
                if not instance_id or semantic not in THING_LABELS:
                    continue
                identifier = f'{scene.name}_{int(panoptic):010d}'
                instance_path = anomaly / 'instances' / f'{identifier}_{scan.stem}.npy'
                instance_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(instance_path, points[labels == panoptic].astype(np.float32))
                entry = instances.setdefault(identifier, dict(semantic_label=THING_LABELS[semantic], filepaths=[]))
                # Both index variants work directly, independently of the reader's cwd.
                entry['filepaths'].append(str(instance_path))
    if extract_instances and not instances:
        raise ValueError('No known object instances were found in the training split')
    metadata.mkdir(parents=True, exist_ok=True)
    with targets[0].open('x') as stream:
        yaml.safe_dump(records, stream, sort_keys=False)
    if extract_instances:
        for target in targets[1:]:
            with target.open('x') as stream:
                yaml.safe_dump(list(instances.values()), stream, sort_keys=False)
    print(f'Wrote {len(records)} scan records and {len(instances)} object tracks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('data/STU'))
    parser.add_argument('--metadata-dir', type=Path, default=Path('data/stu'))
    parser.add_argument('--anomaly-dir', type=Path, default=Path('data/anomaly'))
    parser.add_argument('--metadata-only', action='store_true', help='Use an existing anomaly bank unchanged')
    args = parser.parse_args()
    build(args.dataset_root, args.metadata_dir, args.anomaly_dir, not args.metadata_only)


if __name__ == '__main__':
    main()
