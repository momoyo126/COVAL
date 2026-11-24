"""CPU checks; sparse convolution execution is deliberately not mocked as a validation result."""

import importlib.util
import os
import runpy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# TorchSparse 2.1 queries CUDA capability at import. These tests only construct
# modules and exercise a dense stub; real sparse-kernel checks are separate.
with patch('torch.cuda.get_device_capability', return_value=(8, 0)):
    from pointcept.models import build_model
    from pointcept.models.builder import MODELS
    from pointcept.utils.config import Config
    from pointcept.datasets.anomgen import AnomalyGenerator, SynAnomalyGenerator
    from pointcept.datasets.transform import Compose


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@MODELS.register_module()
class _DenseStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 832)
        self.classifier = torch.nn.Linear(96, 20)
        self.grad_modes = []

    def features(self, data):
        self.grad_modes.append(torch.is_grad_enabled())
        value = self.projection(data['feat'])
        return value + value.mean(0) * 0.1

    def forward(self, data):
        return self.classifier(self.features(data)[:, :96])

    def forward_anom_without_logits(self, data):
        feature = self.features(data)
        widths = (96, 96, 128, 256, 256)
        coords = data['grid_coord']
        layers = [SimpleNamespace(F=part, C=coords) for part in feature.split(widths, dim=1)]
        return layers, feature

    def forward_anom(self, data):
        layers, feature = self.forward_anom_without_logits(data)
        return self.classifier(feature[:, :96]), layers, feature

    def compute_multiscale_consistency_loss(self, full, factual):
        losses = []
        for target, current in zip(full, factual):
            index = (current.C[:, None] == target.C[None]).all(-1).long().argmax(1)
            losses.append(F.mse_loss(current.F, target.F[index]))
        return torch.stack(losses).mean()


def scene():
    coordinate = torch.stack((torch.arange(1, 13), torch.zeros(12), torch.zeros(12)), 1).float()
    return dict(coord=coordinate, grid_coord=coordinate.int(),
                feat=torch.cat((coordinate, torch.ones(12, 1)), 1),
                segment=torch.tensor([8, 19, 8, 8, 8, 8, 8, 19, 8, 8, 8, 8]),
                offset=torch.tensor([6, 12]),
                occlusion_mask=torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1]).bool(),
                anomaly_mask_rv=torch.ones(2, 1, 4, 8).bool())


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_training_configs_inherit_shared_runtime(self):
        for name in ('seg-minknet34', 'seg-minknet34-syn', 'seg-coval'):
            local = runpy.run_path(str(ROOT / 'configs/stu' / f'{name}.py'))
            self.assertEqual(local['_base_'], ['../_base_/default_runtime.py'])
            for field in ('resume', 'seed', 'batch_size', 'num_worker', 'enable_amp', 'train', 'test'):
                self.assertNotIn(field, local)
        baseline = Config.fromfile(str(ROOT / 'configs/stu/seg-minknet34-syn.py'))
        self.assertTrue(baseline.hooks[0].fune_tuning)
        coval = Config.fromfile(str(ROOT / 'configs/stu/seg-coval.py'))
        self.assertNotIn('fune_tuning', coval.hooks[0])

    def test_configs_and_registry(self):
        for name, epochs in [('seg-minknet34', 50), ('seg-minknet34-syn', 5), ('seg-coval', 3)]:
            cfg = Config.fromfile(str(ROOT / 'configs/stu' / f'{name}.py'))
            self.assertEqual(cfg.seed, 13256645)
            self.assertEqual(cfg.epoch, epochs)
            self.assertEqual(cfg.batch_size, 8)
            self.assertEqual(cfg.num_worker, 8)
            self.assertFalse(cfg.enable_swanlab)
        for name in ('test-minknet34-syn', 'test-coval'):
            cfg = Config.fromfile(str(ROOT / 'configs/stu' / f'{name}.py'))
            self.assertEqual(cfg.data.test.split, 'test')
            self.assertEqual(cfg.test.type, 'AnomSegTester')
            self.assertFalse(cfg.test.save_conf)
        syn = Config.fromfile(str(ROOT / 'configs/stu/seg-minknet34-syn.py'))
        self.assertEqual(syn.data.train.transform[0].type, 'SynAnomalyGenerator')
        coval = Config.fromfile(str(ROOT / 'configs/stu/seg-coval.py'))
        self.assertEqual(coval.save_path, 'exp/stu/coval')
        self.assertEqual(coval.weight, 'exp/stu/minknet34-syn/model/model_last.pth')
        coval_test = Config.fromfile(str(ROOT / 'configs/stu/test-coval.py'))
        self.assertEqual(coval_test.save_path, 'exp/stu/coval')
        self.assertEqual(coval_test.weight, 'exp/stu/coval/model/model_best.pth')
        self.assertEqual(MODELS.get('Minknet34-ts').__name__, 'Minknet34')
        self.assertNotIn('Minknet34-GHPA-ExplicitHighPass', MODELS.module_dict)

    def test_baseline_does_not_apply_factual_mask(self):
        model = build_model(dict(type='AnomSegmentor', backbone=dict(type='_DenseStub'),
                                 criteria=[dict(type='CrossEntropyLoss', ignore_index=-1)])).train()
        data = scene()
        original = data['segment'].clone()
        loss = model(data)['loss']
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(data['segment'], original)
        self.assertEqual(model.backbone.grad_modes, [True])
        loss.backward()
        self.assertIsNotNone(model.backbone.projection.weight.grad)

    def test_complete_method_uses_factual_ce_and_no_grad_target(self):
        from pointcept.models.anomseg.anomsegmentor import LaserScan
        model = build_model(dict(type='AnomSegmentor', backbone=dict(type='_DenseStub'),
            criteria=[dict(type='CrossEntropyLoss', ignore_index=-1)],
            recon=dict(type='TinyFeatureReconNet', in_channels=832, latent_ratio=0.25),
            mutil_scale_consis=True)).train()
        model.H, model.W = 4, 8
        model.range_project = LaserScan(H=4, W=8, fov_up=22.5, fov_down=-22.5)
        data = scene()
        expected = data['segment'][data['occlusion_mask']].clone()
        loss = model(data)['loss']
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(data['segment'], expected)
        self.assertEqual(model.backbone.grad_modes, [True, False])
        self.assertIn('loss/loss_recon', model.swanlab_results)
        self.assertIn('loss/loss_mutil_scale_consis', model.swanlab_results)
        loss.backward()
        self.assertGreater(model.recon_head.linera1.weight.grad.abs().sum().item(), 0)

    def test_generator_database_names_and_portable_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'instances').mkdir()
            np.save(root / 'instances' / 'object.npy', np.ones((30, 4)))
            entry = [dict(semantic_label=1, filepaths=['data/instances/object.npy'])]
            (root / 'train_instances_database.yaml').write_text(yaml.safe_dump(entry))
            # Official-only metadata works for both training methods.
            for kind in (AnomalyGenerator, SynAnomalyGenerator):
                generator = kind(database_path=root)
                self.assertEqual(generator.instance_data, entry)
                self.assertEqual(generator.select_random_instance().shape, (30, 4))
            custom = [dict(semantic_label=2, filepaths=['data/stu/instances/object.npy'])]
            (root / 'train_instances_database_pointcept.yaml').write_text(yaml.safe_dump(custom))
            generator = AnomalyGenerator(root)
            self.assertEqual(generator.instance_data, custom)
            self.assertEqual(generator.select_random_instance().shape, (30, 4))
            self.assertEqual(SynAnomalyGenerator(root).instance_data, entry)
            self.assertEqual(AnomalyGenerator(root, instance_database='train_instances_database.yaml').instance_data, entry)
            with self.assertRaises(FileNotFoundError):
                AnomalyGenerator(root, instance_database='missing.yaml')
            for path in ('instances/object.npy', 'data/instances/object.npy',
                         'data/stu/instances/object.npy', '/old/location/instances/object.npy',
                         str(root / 'instances' / 'object.npy')):
                self.assertEqual(generator.resolve_instance_path(path), root / 'instances' / 'object.npy')
            with self.assertRaises(FileNotFoundError):
                generator.resolve_instance_path('instances/missing.npy')
            raw = dict(coord=np.zeros((2, 3)))
            with patch.object(SynAnomalyGenerator, 'create_anomaly_scene', return_value=raw):
                self.assertIs(SynAnomalyGenerator(root)(raw), raw)

    def test_probability_files_are_not_softmaxed_again(self):
        metrics = load_tool('compute_anom_metric')
        self.assertIsNone(metrics.calculate_ood_metrics([], [])['AUROC'])
        self.assertIsNone(metrics.calculate_ood_metrics([np.zeros(2)], [np.zeros(2)])['AUROC'])
        logits = np.zeros((3, 20), dtype=np.float32)
        logits[:, 19] = [-4, 0, 4]
        probabilities = metrics.softmax_np(logits)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'frame.npy'
            np.save(path, probabilities.astype(np.float16))
            np.testing.assert_array_equal(metrics.scores_from_conf(path, 'anomlogit'),
                                          probabilities[:, 19].astype(np.float16))
            np.save(path, logits)
            np.testing.assert_array_equal(metrics.scores_from_conf(path, 'anomlogit', 'logits'),
                                          probabilities[:, 19].astype(np.float16))

    def test_scalar_anomaly_scores_are_used_directly(self):
        metrics = load_tool('compute_anom_metric')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / '000001.npy'
            scores = np.array([0, 0.1, 0.5, 0.9, 1], dtype=np.float16)
            np.save(path, scores)
            np.testing.assert_array_equal(metrics.scores_from_conf(path, 'anomlogit'), scores)
            with self.assertRaises(ValueError):
                metrics.scores_from_conf(path, 'maxlogit')
            for invalid in (np.array([np.nan]), np.array([-0.1]), np.array([1.1])):
                np.save(path, invalid)
                with self.assertRaises(ValueError):
                    metrics.scores_from_conf(path, 'anomlogit')

    def test_inference_saves_sequence_anomaly_scores(self):
        from pointcept.engines.test import AnomSegTester
        from unittest.mock import Mock

        class Loader(list):
            batch_size = 1

        logits = torch.arange(40).reshape(2, 20).float() / 10
        inverse = np.array([0, 1, 1, 0])
        fragment = dict(index=torch.arange(2), coord=torch.zeros(2, 3), feat=torch.ones(2, 4),
                        offset=torch.tensor([2]))
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(save_path=temporary, empty_cache=False, resume=False,
                                     data=SimpleNamespace(num_classes=20))
            loader = Loader([[dict(fragment_list=[fragment], segment=np.zeros(2),
                name='125_000001', origin_segment=np.zeros(4), inverse=inverse)]])
            tester = AnomSegTester(cfg=config, model=Mock(return_value=dict(seg_logits=logits)),
                                   test_loader=loader)
            self.assertFalse(tester.save_conf)
            with patch.object(torch.Tensor, 'cuda', lambda value, *args, **kwargs: value):
                tester.test()
            directory = Path(temporary) / 'conf'
            self.assertEqual({path.name for path in directory.iterdir()},
                             {'sequences_prediction', 'sequences_anomscore'})
            prediction = np.load(directory / 'sequences_prediction/125/000001.npy')
            self.assertEqual(prediction.dtype, np.dtype('uint8'))
            np.testing.assert_array_equal(prediction, logits.argmax(1).numpy()[inverse])
            actual = np.load(directory / 'sequences_anomscore/125/000001.npy')
            expected = logits.softmax(1).numpy().astype(np.float16)[inverse, 19]
            np.testing.assert_array_equal(actual, expected)

    def test_metadata_generation_and_no_overwrite(self):
        from pointcept.datasets.stu import STUDataset

        prepare = load_tool('prepare_stu')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_dir = root / 'STU/train/00'
            (scene_dir / 'velodyne').mkdir(parents=True)
            (scene_dir / 'labels').mkdir()
            identity = '1 0 0 0 0 1 0 0 0 0 1 0\n'
            (scene_dir / 'calib.txt').write_text('Tr: ' + identity)
            (scene_dir / 'poses.txt').write_text(identity)
            np.ones((30, 4), dtype=np.float32).tofile(scene_dir / 'velodyne/000000.bin')
            np.full(30, (1 << 16) + 10, dtype=np.uint32).tofile(scene_dir / 'labels/000000.label')
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, 'argv', ['prepare_stu.py', '--dataset-root', 'STU']):
                    prepare.main()
            finally:
                os.chdir(previous)
            metadata, anomaly = root / 'data/stu', root / 'data/anomaly'
            records = yaml.safe_load((metadata / 'train_database.yaml').read_text())
            self.assertEqual(len(records), 1)
            for field in ('filepath', 'label_filepath'):
                self.assertTrue(Path(records[0][field]).is_absolute())
                self.assertTrue(Path(records[0][field]).is_file())
            first = (anomaly / 'train_instances_database.yaml').read_bytes()
            second = (anomaly / 'train_instances_database_pointcept.yaml').read_bytes()
            self.assertEqual(first, second)
            for entry in yaml.safe_load(first):
                for path in entry['filepaths']:
                    self.assertTrue(Path(path).is_absolute())
                    np.testing.assert_array_equal(np.load(path), np.ones((30, 4)))
            # Consume the generated files from a different working directory.
            dataset = STUDataset(data_root=str(metadata), split='train')
            np.testing.assert_array_equal(dataset.get_data(0)['coord'], np.ones((30, 3)))
            for kind in (AnomalyGenerator, SynAnomalyGenerator):
                generator = kind(database_path=anomaly)
                np.testing.assert_array_equal(generator.select_random_instance(), np.ones((30, 4)))
            with self.assertRaises(FileExistsError):
                prepare.build(root / 'STU', metadata, anomaly)


if __name__ == '__main__':
    unittest.main()
