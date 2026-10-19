"""Data/metric regression tests; no CUDA or MMDetection dependency."""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('pnxk_converter', 'tools/data_converter/pnxk_converter.py')
m = load('pnxk_metrics', 'mmdet3d/datasets/pnxk_metrics.py')


class PNXKTest(unittest.TestCase):
    def test_converter_multiple_recordings_and_chunks(self):
        import laspy
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = ['20260228_1105_VF6_01_1_2', '20260301_1105_VF6_01_3_4']
            val = ['20260302_1105_VF6_01_5_6']
            transform = np.eye(4); transform[:3, 3] = [1, 2, 3]
            (root / 'VF6_01_Extrinsics.json').write_text(json.dumps({'LIDAR_TOP': transform.tolist()}))
            for recording in train + val:
                for chunk in ('chunk-a', 'chunk-b'):
                    cloud_dir = root / 'data' / recording / chunk / 'LIDAR' / 'LIDAR_TOP'
                    label_dir = root / 'label' / recording / chunk
                    cloud_dir.mkdir(parents=True); label_dir.mkdir(parents=True)
                    cloud = laspy.LasData(laspy.LasHeader(point_format=0, version='1.2'))
                    cloud.x = [0., 1.]; cloud.y = [0., 1.]; cloud.z = [0., 1.]
                    cloud.intensity = [10, 20]
                    cloud.write(cloud_dir / '100-000000000.laz')
                    (label_dir / '100-000000000.txt').write_text('1 2 3 4 2 1 0 Car 1 2 7\n')
            (root / 'train.txt').write_text('\n'.join('old/data/' + r for r in train))
            (root / 'val.txt').write_text('\n'.join(val))
            args = argparse.Namespace(root=str(root), output=str(root / 'cache'),
                train_split=str(root / 'train.txt'), val_split=str(root / 'val.txt'),
                with_cameras=False, camera_extrinsics=None, distortion_model=None,
                allow_split_overlap=False, skip_missing_labels=False, max_frames_per_chunk=0,
                lidar_max_delta_ms=1., camera_max_delta_ms=40., skip_unmatched=False)
            c.convert(args)
            for split, expected in [('train', train), ('val', val)]:
                with (root / 'cache' / ('pnxk_infos_' + split + '.pkl')).open('rb') as handle:
                    data = pickle.load(handle)
                self.assertEqual(len(data['infos']), len(expected) * 2)
                self.assertEqual({i['recording'] for i in data['infos']}, set(expected))
                info = data['infos'][0]
                self.assertIsInstance(info['gt_boxes'], list)
                points = np.fromfile(root / 'cache' / info['lidar_path'], dtype=np.float32).reshape(-1, 5)
                np.testing.assert_allclose(points, [[1, 2, 3, 10, 0], [2, 3, 4, 20, 0]])

    def test_timestamp_precision_and_threshold(self):
        self.assertEqual(c.timestamp_ns('1772252610-099740267.txt'), 1772252610099740267)
        index = ([1000000000, 1040000000], [Path('a'), Path('b')])
        self.assertEqual(c.nearest(index, 1039000000, 2), (Path('b'), 1.0))
        with self.assertRaises(ValueError):
            c.nearest(index, 1020000000, 2)

    def test_recording_split_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data' / 'recording').mkdir(parents=True)
            split = root / 'split.txt'
            split.write_text('old-prefix/data/recording\n')
            self.assertEqual(c.read_split(split, root), ['recording'])
            args = argparse.Namespace(root=str(root), output=str(root / 'cache'),
                train_split=str(split), val_split=str(split), with_cameras=False,
                camera_extrinsics=None, distortion_model=None, allow_split_overlap=False)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                c.convert(args)
            self.assertFalse((root / 'cache').exists())
            split.write_text('recording\nrecording\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                c.read_split(split, root)

    def test_labels_preserve_geometric_center(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'label.txt'
            path.write_text('1 2 3 4 2 1.5 1.57 Rider 3 0 10\n')
            info = c.read_labels(path)
            np.testing.assert_allclose(info['gt_boxes'][0, :3], [1, 2, 3])
            self.assertEqual(info['gt_names'][0], 'rider')
            self.assertEqual(info['annotation_num_points'][0], 0)
            path.write_text('1 2 3 -4 2 1 0 Car 1 1 0\n')
            with self.assertRaises(ValueError):
                c.read_labels(path)

    def test_date_calibration_has_no_silent_fallback(self):
        root = ROOT / 'Data-PNXK'
        if not root.exists():
            self.skipTest('Sample data is local only')
        ext, intr = c.calibration(root, '20260228_1105_VF6_01_1772251511_1772254880', c.CAMERAS)
        self.assertEqual(np.asarray(ext['CAM_P_F']).shape, (4, 4))
        self.assertIn('CAM_P_F', intr)
        with self.assertRaises(ValueError):
            c.calibration(root, '20260710_1714_VF6_01_1_2', c.CAMERAS)

    def test_metric_one_to_one_and_class_matching(self):
        gt = [dict(centers=np.array([[0., 0., 0.], [10., 0., 0.]]), labels=np.array([0, 0]))]
        pred = [dict(centers=np.array([[0., 0., 0.], [0., 0., 0.], [10., 0., 0.]]),
                     labels=np.array([0, 0, 0]), scores=np.array([.9, .8, .7]))]
        metrics = m.center_distance_ap(gt, pred, ['car'], [1.])
        self.assertAlmostEqual(metrics['center_mAP@1.0m'], 5 / 6)
        pred[0]['labels'][:] = 1
        self.assertEqual(m.center_distance_ap(gt, pred, ['car'], [1.])['center_mAP@1.0m'], 0)
        perfect = [dict(centers=gt[0]['centers'], labels=gt[0]['labels'], scores=np.ones(2))]
        self.assertEqual(m.center_distance_ap(gt, perfect, ['car'], [1.])['center_mAP@1.0m'], 1)


if __name__ == '__main__':
    unittest.main()
