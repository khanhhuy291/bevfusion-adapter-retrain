"""Validate a converted PNXK cache on CPU; optionally build a real pipeline batch."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def check_cache(root):
    from PIL import Image

    report = {}
    for split in ('train', 'val'):
        with (root / ('pnxk_infos_' + split + '.pkl')).open('rb') as handle:
            data = pickle.load(handle)
        assert data['metadata']['schema_version'] == 1
        assert data['metadata']['point_frame'] == 'ego'
        assert data['metadata']['box_origin'] == 'geometric_center'
        assert data['infos'], 'Empty split'
        tokens = set()
        for info in data['infos']:
            assert info['token'] not in tokens
            tokens.add(info['token'])
            assert info['recording'] in data['metadata']['recordings']
            points = np.fromfile(root / info['lidar_path'], dtype=np.float32).reshape(-1, 5)
            assert len(points) == info['num_points'] and np.isfinite(points).all()
            assert np.all(points[:, 4] == 0), 'Expected single-sweep time lag'
            boxes = np.asarray(info['gt_boxes'], dtype=np.float32).reshape(-1, 7)
            assert boxes.shape == (len(info['gt_names']), 7)
            assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
            assert set(info['gt_names']) <= set(data['metadata']['classes'])
            assert [c['name'] for c in info['cams']] == data['metadata']['cameras']
            for camera in info['cams']:
                t = np.asarray(camera['ego2camera'])
                assert np.allclose(t @ np.linalg.inv(t), np.eye(4), atol=1e-5)
                assert np.asarray(camera['intrinsic']).shape == (3, 3)
                with Image.open(root / camera['data_path']) as image:
                    image.verify()
        report[split] = dict(samples=len(tokens), recordings=len(data['metadata']['recordings']),
                             cameras=len(data['metadata']['cameras']),
                             split_overlap=data['metadata']['split_overlap'])
    print(json.dumps(report, indent=2))


def check_pipeline(config_path, root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from torchpack.utils.config import configs
    from mmcv import Config
    from mmcv.parallel import collate
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset

    configs.load(config_path, recursive=True)
    configs.update({'dataset_root': str(root) + '/'})
    cfg = Config(recursive_eval(configs))
    for split in ('train', 'val'):
        dataset = build_dataset(cfg.data[split])
        raw = dataset.get_data_info(0)
        expected_centers = dataset.data_infos[0]['gt_boxes'][:, :3]
        np.testing.assert_allclose(raw['ann_info']['gt_bboxes_3d'].gravity_center.numpy(),
                                   expected_centers, atol=1e-5)
        np.testing.assert_allclose(raw['lidar2ego'], np.eye(4), atol=1e-6)
        if cfg.use_camera:
            expected_cameras = ['CAM_P_F', 'CAM_P_FL', 'CAM_P_FR',
                                'CAM_P_B', 'CAM_P_LB', 'CAM_P_RB']
            assert dataset.metadata['cameras'] == expected_cameras
            assert len(raw['image_paths']) == 6
            for i in range(6):
                np.testing.assert_allclose(raw['camera2lidar'][i] @ raw['lidar2camera'][i],
                                           np.eye(4), atol=1e-5)
                np.testing.assert_allclose(raw['camera2ego'][i], raw['camera2lidar'][i])
                np.testing.assert_allclose(raw['lidar2image'][i],
                                           raw['camera_intrinsics'][i] @ raw['lidar2camera'][i])
        sample = dataset[0]
        required = {'img', 'points', 'gt_bboxes_3d', 'gt_labels_3d', 'metas', 'depths',
                    'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image',
                    'camera_intrinsics', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix'}
        assert required <= set(sample), required - set(sample)
        assert sample['points'].data.shape[1] == 5
        assert sample['gt_bboxes_3d'].data.tensor.shape[1] == 7
        assert len(sample['gt_bboxes_3d'].data) > 0
        if cfg.use_camera:
            assert tuple(sample['img'].data.shape) == (6, 3, *cfg.image_size)
            for key in ('camera2ego', 'lidar2camera', 'lidar2image', 'camera_intrinsics',
                        'camera2lidar', 'img_aug_matrix'):
                assert tuple(sample[key].data.shape) == (6, 4, 4), key
            vtransform = cfg.model.encoders.camera.vtransform
            factor = cfg.model.heads.object.test_cfg.out_size_factor
            for axis, bound in enumerate((vtransform.xbound, vtransform.ybound)):
                camera_bins = round((bound[1] - bound[0]) / bound[2] / vtransform.downsample)
                lidar_bins = cfg.model.heads.object.test_cfg.grid_size[axis] // factor
                assert camera_bins == lidar_bins, 'Camera/LiDAR BEV resolutions differ'
        batch = collate([sample], samples_per_gpu=1)
        print('{} pipeline/collate OK: {} points, img {}'.format(
            split, len(sample['points'].data), tuple(sample['img'].data.shape)))
        assert 'metas' in batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/pnxk')
    parser.add_argument('--pipeline-config', help='Requires the original mmcv/mmdet training environment')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    check_cache(root)
    if args.pipeline_config:
        check_pipeline(args.pipeline_config, root)


if __name__ == '__main__':
    main()
