"""PNXK single-sweep dataset. All cached points and boxes use ego axes."""
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmdet.datasets import DATASETS
from mmdet.datasets.builder import PIPELINES

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines.formating import Collect3D
from .pnxk_metrics import center_distance_ap


@DATASETS.register_module()
class PNXKDataset(Custom3DDataset):
    CLASSES = ('car', 'rider', 'pedestrian')

    def __init__(self, point_cloud_range=None, **kwargs):
        self.point_cloud_range = point_cloud_range
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        self.metadata = data['metadata']
        if self.metadata.get('schema_version') != 1 or self.metadata.get('point_frame') != 'ego':
            raise ValueError('Unsupported PNXK cache schema or point frame')
        if tuple(self.metadata['classes']) != tuple(self.CLASSES):
            raise ValueError('PNXK class order must match converter metadata')
        if self.modality.get('use_camera') and not self.metadata['cameras']:
            raise ValueError('Fusion requires a cache generated with --with-cameras')
        for info in data['infos']:
            info['gt_boxes'] = np.asarray(info['gt_boxes'], dtype=np.float32).reshape(-1, 7)
            info['gt_names'] = np.asarray(info['gt_names'])
        return data['infos']

    def get_cat_ids(self, idx):
        return sorted({self.cat2id[n] for n in self.data_infos[idx]['gt_names']})

    def get_ann_info(self, index):
        info = self.data_infos[index]
        names = info['gt_names'].copy()
        # Constructor converts geometric center to the framework's bottom center.
        boxes = LiDARInstance3DBoxes(info['gt_boxes'].copy(), box_dim=7,
                                    origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        return dict(gt_bboxes_3d=boxes,
                    gt_labels_3d=np.asarray([self.cat2id[n] for n in names], dtype=np.int64),
                    gt_names=names)

    def get_data_info(self, index):
        info = self.data_infos[index]
        data = dict(token=info['token'], sample_idx=info['token'], timestamp=info['timestamp'],
                    lidar_path=str(Path(self.dataset_root) / info['lidar_path']), sweeps=[],
                    ego2global=np.eye(4, dtype=np.float32),
                    # The cached "lidar" frame is already ego, not the raw TOP frame.
                    lidar2ego=np.eye(4, dtype=np.float32),
                    img_aug_matrix=[np.eye(4, dtype=np.float32)],
                    lidar_aug_matrix=np.eye(4, dtype=np.float32))
        for key in ('camera_intrinsics', 'camera2ego', 'lidar2camera', 'lidar2image', 'camera2lidar'):
            data[key] = []
        data['image_paths'] = []
        if self.modality.get('use_camera'):
            for camera in info['cams']:
                t = np.asarray(camera['ego2camera'], dtype=np.float32)
                k = np.eye(4, dtype=np.float32)
                k[:3, :3] = camera['intrinsic']
                data['image_paths'].append(str(Path(self.dataset_root) / camera['data_path']))
                data['camera_intrinsics'].append(k)
                data['camera2ego'].append(np.linalg.inv(t).astype(np.float32))
                data['lidar2camera'].append(t)
                data['lidar2image'].append(k @ t)
                data['camera2lidar'].append(np.linalg.inv(t).astype(np.float32))
        else:
            # Supply the common BEVFusion forward signature, unused by lidar encoder.
            for key in ('camera_intrinsics', 'camera2ego', 'lidar2camera', 'lidar2image', 'camera2lidar'):
                data[key] = [np.eye(4, dtype=np.float32)]
        data['ann_info'] = self.get_ann_info(index)
        return data

    def evaluate(self, results, metric='center_distance', logger=None, **kwargs):
        if isinstance(metric, (list, tuple)) and len(metric) == 1:
            metric = metric[0]
        if metric != 'center_distance':
            raise ValueError('PNXK supports diagnostic center_distance metric only')
        if len(results) != len(self):
            raise ValueError('Prediction count differs from dataset length')
        ground_truth, predictions = [], []
        def in_range(centers):
            if self.point_cloud_range is None:
                return np.ones(len(centers), dtype=bool)
            r = np.asarray(self.point_cloud_range)
            return ((centers >= r[:3]) & (centers < r[3:])).all(axis=1)
        for i, pred in enumerate(results):
            info = self.data_infos[i]
            centers = info['gt_boxes'][:, :3]
            mask = in_range(centers)
            ground_truth.append(dict(centers=centers[mask], labels=np.asarray(
                [self.cat2id[n] for n in info['gt_names']], dtype=np.int64)[mask]))
            centers = pred['boxes_3d'].gravity_center.detach().cpu().numpy()
            mask = in_range(centers)
            predictions.append(dict(centers=centers[mask],
                                    scores=pred['scores_3d'].detach().cpu().numpy()[mask],
                                    labels=pred['labels_3d'].detach().cpu().numpy()[mask]))
        return center_distance_ap(ground_truth, predictions, self.CLASSES)


@PIPELINES.register_module()
class PNXKCollect3D(Collect3D):
    """Fill unused camera/depth arguments without loading dummy sensor files."""
    def __call__(self, results):
        data = super().__call__(results)
        if 'img' not in data:
            data['img'] = DataContainer(torch.empty((0, 3, 1, 1)), stack=True)
        # Neither DepthLSSTransform nor lidar-only uses supervised depth targets.
        data['depths'] = DataContainer(torch.empty((0,)), stack=True, pad_dims=None)
        return data
