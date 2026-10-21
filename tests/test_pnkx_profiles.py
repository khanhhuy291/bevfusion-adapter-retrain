"""Profile and geometry checks using the real legacy runtime (no GPU needed)."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

RUNTIME = importlib.util.find_spec('mmcv') is not None and importlib.util.find_spec('torchpack') is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(RUNTIME, 'Run inside the BEVFusion Docker image')
class PNKXProfilesTest(unittest.TestCase):
    def test_reference_matches_nuscenes_model_except_adaptations(self):
        import sys
        sys.path.insert(0, str(ROOT / 'tools'))
        from compare_pnkx_configs import resolve, differences
        base = resolve('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml')
        ref = resolve('configs/pnkx/nuscenes/fusion.yaml')
        allowed = {'model.encoders.camera.backbone.init_cfg.checkpoint',
                   'model.heads.object.num_classes', 'model.heads.object.common_heads.vel',
                   'model.heads.object.bbox_coder.code_size',
                   'model.heads.object.train_cfg.dataset', 'model.heads.object.test_cfg.dataset',
                   'model.heads.object.train_cfg.code_weights', 'model.heads.map'}
        found = {d['path'] for d in differences(ref['model'], base['model'])}
        self.assertLessEqual(found, allowed)
        self.assertEqual(ref['max_epochs'], 6)
        self.assertEqual(ref['optimizer'], base['optimizer'])
        self.assertEqual(ref['lr_config'], base['lr_config'])
        self.assertEqual(ref['fp16'], base['fp16'])
        self.assertEqual(ref['data']['samples_per_gpu'], 4)
        self.assertEqual(ref['data']['train']['type'], 'CBGSDataset')
        self.assertEqual(ref['data']['train']['dataset']['type'], 'PNKXDataset')
        self.assertTrue(ref['model']['encoders']['camera']['backbone']['init_cfg']['checkpoint'])
        self.assertEqual(ref['load_from'], 'pretrained/lidar-only-det.pth')
        transforms = {t['type']: t for t in ref['train_pipeline']}
        self.assertTrue(transforms['GlobalRotScaleTrans']['is_train'])
        self.assertTrue(transforms['ImageAug3D']['is_train'])
        self.assertIn('RandomFlip3D', transforms)
        for unsafe in ['LoadRadarPointsMultiSweeps', 'ObjectPaste', 'LoadPointsFromMultiSweeps', 'GTDepth']:
            self.assertNotIn(unsafe, transforms)

    def test_8gb_and_lidar_profiles(self):
        import sys
        sys.path.insert(0, str(ROOT / 'tools'))
        from compare_pnkx_configs import resolve
        for mode in ('fusion', 'lidar'):
            cfg = resolve('configs/pnkx/nuscenes/' + mode + '_8gb.yaml')
            self.assertEqual(cfg['data']['samples_per_gpu'], 1)
            self.assertEqual(cfg['optimizer_config']['cumulative_iters'], 4)
            self.assertEqual(cfg['point_cloud_range'], [-54, -54, -5, 54, 54, 3])
            self.assertEqual(cfg['voxel_size'], [.075, .075, .2])
            self.assertEqual(cfg['model']['heads']['object']['test_cfg']['grid_size'], [1440, 1440, 41])
            if mode == 'fusion':
                self.assertTrue(cfg['model']['encoders']['camera']['backbone']['with_cp'])
                self.assertEqual(cfg['max_epochs'], 6)
            else:
                self.assertIsNone(cfg['model']['encoders']['camera'])
                self.assertIsNone(cfg['model']['fuser'])
                self.assertEqual(cfg['max_epochs'], 20)
                self.assertEqual(cfg['optimizer']['lr'], 1e-4)
                self.assertEqual(cfg['lr_config']['policy'], 'cyclic')

    def test_real_box_corners_and_augmentations(self):
        import torch
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.core.points import LiDARPoints
        from mmdet3d.datasets.pnkx_geometry import raw_boxes_to_internal
        from mmdet3d.datasets.pipelines.transforms_3d import GlobalRotScaleTrans, RandomFlip3D
        raw = np.array([[7., 3., 1., 4., 1.5, 2., .4]], dtype=np.float32)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        local = signs * raw[0, 3:6] / 2
        c, s = np.cos(raw[0, 6]), np.sin(raw[0, 6])
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        physical = local @ rotation.T + raw[0, :3]
        def same_points(a, b):
            distances = np.linalg.norm(np.asarray(a)[:, None] - np.asarray(b)[None], axis=-1)
            self.assertLess(float(distances.min(axis=1).max()), 1e-4)
            self.assertLess(float(distances.min(axis=0).max()), 1e-4)
        boxes = LiDARInstance3DBoxes(raw_boxes_to_internal(raw), box_dim=7, origin=(.5, .5, .5))
        same_points(boxes.corners[0].numpy(), physical)
        data = dict(points=LiDARPoints(physical.astype(np.float32), points_dim=3), gt_bboxes_3d=boxes)
        np.random.seed(4)
        transform = GlobalRotScaleTrans(resize_lim=[.9, 1.1],rot_lim=[-.7, .7],trans_lim=.5,is_train=True)
        data = transform(data)
        matrix = data['lidar_aug_matrix']
        expected = physical @ matrix[:3, :3].T + matrix[:3, 3]
        same_points(data['points'].tensor.numpy(), expected)
        same_points(data['gt_bboxes_3d'].corners[0].numpy(), expected)
        for _ in range(8):
            data = RandomFlip3D()(data)
            matrix = data['lidar_aug_matrix']
            expected = physical @ matrix[:3, :3].T + matrix[:3, 3]
            same_points(data['points'].tensor.numpy(), expected)
            same_points(data['gt_bboxes_3d'].corners[0].numpy(), expected)


if __name__ == '__main__':
    unittest.main()
