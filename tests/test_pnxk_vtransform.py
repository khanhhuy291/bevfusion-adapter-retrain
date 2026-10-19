"""CPU regression for DepthLSS's interface with the repository's depth base.

Only registry/decorators and CUDA pooling are substituted; actual Torch depth
projection, geometry and convolution code run on CPU.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


class DepthLSSInterfaceTest(unittest.TestCase):
    def test_single_depth_channel_and_base_call(self):
        root = Path(__file__).resolve().parents[1]
        registry = types.SimpleNamespace(register_module=lambda: lambda cls: cls)
        def force_fp32(*args, **kwargs):
            return lambda fn: fn
        stubs = {
            'mmcv': types.ModuleType('mmcv'),
            'mmcv.runner': types.SimpleNamespace(force_fp32=force_fp32),
            'mmdet3d': types.ModuleType('mmdet3d'),
            'mmdet3d.ops': types.SimpleNamespace(bev_pool=None),
            'mmdet3d.models': types.ModuleType('mmdet3d.models'),
            'mmdet3d.models.builder': types.SimpleNamespace(VTRANSFORMS=registry),
            'mmdet3d.models.vtransforms': types.ModuleType('mmdet3d.models.vtransforms'),
        }
        with patch.dict(sys.modules, stubs):
            modules = {}
            for short in ('base', 'depth_lss'):
                name = 'mmdet3d.models.vtransforms.' + short
                spec = importlib.util.spec_from_file_location(name, root / 'mmdet3d/models/vtransforms' / (short + '.py'))
                module = importlib.util.module_from_spec(spec)
                # patch.dict restores these entries too, leaving no import stubs behind.
                with patch.dict(sys.modules, {name: module}):
                    spec.loader.exec_module(module)
                modules[name] = module
                sys.modules[name] = module
            model = modules['mmdet3d.models.vtransforms.depth_lss'].DepthLSSTransform(
                in_channels=8, out_channels=4, image_size=(32, 32), feature_size=(4, 4),
                xbound=(-4, 4, 1), ybound=(-4, 4, 1), zbound=(-2, 2, 4), dbound=(1, 4, 1))
            self.assertFalse(model.height_expand)
            self.assertFalse(model.add_depth_features)
            def cpu_pool(geom, features):
                self.assertEqual(tuple(geom.shape), (1, 1, 3, 4, 4, 3))
                return features.sum(dim=(1, 2)).permute(0, 3, 1, 2)
            model.bev_pool = cpu_pool
            model.eval()
            views = torch.eye(4).reshape(1, 1, 4, 4)
            ego = torch.eye(4).reshape(1, 4, 4)
            # First two points hit one pixel after undoing augmentation. The
            # nearest depth must win; a point behind the camera must be absent.
            ego[:, 0, 3] = 1.
            points = [torch.tensor([[3., 1., 2., 50., 0.],
                                    [5., 2., 4., 60., 0.],
                                    [1., 0., -1., 70., 0.]])]
            original = points[0].clone()
            original_get_cam_feats = model.get_cam_feats
            def checked_feats(x, depth, mats):
                self.assertEqual(int(torch.count_nonzero(depth)), 1)
                self.assertEqual(float(depth[0, 0, 0, 0, 1]), 2.)
                return original_get_cam_feats(x, depth, mats)
            model.get_cam_feats = checked_feats
            with torch.no_grad():
                output = model(torch.ones(1, 1, 8, 4, 4), points, None,
                               views, ego, views, views, views, views, views, ego, [{}])
            self.assertEqual(tuple(output.shape), (1, 4, 4, 4))
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(points[0], original)
            with self.subTest('BEV lower bound must exclude points below range'):
                base = modules['mmdet3d.models.vtransforms.base']
                captured = {}
                def fake_cuda_pool(features, coordinates, batch, depth_bins, x_bins, y_bins):
                    captured['coordinates'] = coordinates
                    return torch.zeros(batch, 4, depth_bins, x_bins, y_bins)
                base.bev_pool = fake_cuda_pool
                # xbound starts at -4. The first point is outside; the second
                # is inside voxel x=0. Integer truncation used to keep both.
                geometry = torch.tensor([[-4.1, 0., 0.], [-3.9, 0., 0.]]).reshape(1, 1, 1, 1, 2, 3)
                lifted = torch.ones(1, 1, 1, 1, 2, 4)
                pooled = base.BaseTransform.bev_pool(model, geometry, lifted)
                self.assertEqual(len(captured['coordinates']), 1)
                self.assertEqual(int(captured['coordinates'][0, 0]), 0)
                self.assertEqual(tuple(pooled.shape), (1, 4, 8, 8))


if __name__ == '__main__':
    unittest.main()
