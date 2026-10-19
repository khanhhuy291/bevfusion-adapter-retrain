"""One real GPU forward/backward to audit both branches of PNXK fusion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet3d.utils.torch_compat import patch_mmcv_scatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pnxk/fusion.yaml')
    parser.add_argument('--output', default='outputs/pnxk-fusion-audit.json')
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    patch_mmcv_scatter()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs))
    assert cfg.use_camera, 'This audit requires fusion'
    dataset = build_dataset(cfg.data.train)
    batch = scatter(collate([dataset[0]], samples_per_gpu=1), [0])[0]
    assert tuple(batch['img'].shape) == (1, 6, 3, *cfg.image_size)
    before = [p.clone() for p in batch['points']]
    model = build_model(cfg.model).cuda().train()
    model.init_weights()
    report = dict(image_shape=list(batch['img'].shape), feature_shapes={})
    def record(name):
        def hook(module, inputs, output):
            report['feature_shapes'][name] = list(output.shape)
        return hook
    handles = [model.encoders['camera']['vtransform'].register_forward_hook(record('camera_bev')),
               model.encoders['lidar']['backbone'].register_forward_hook(record('lidar_bev')),
               model.fuser.register_forward_hook(record('fused_bev'))]
    output = model(**batch)
    loss = sum(value for key, value in output.items() if key.startswith('loss/'))
    assert torch.isfinite(loss), 'Non-finite loss'
    loss.backward()
    for original, current in zip(before, batch['points']):
        torch.testing.assert_close(current, original, rtol=0, atol=0)
    shapes = report['feature_shapes']
    assert shapes['camera_bev'] == [1, 80, 128, 128]
    assert shapes['lidar_bev'] == shapes['fused_bev'] == [1, 256, 128, 128]
    assert list(model.encoders) == ['camera', 'lidar'], 'Fuser input channel order changed'
    branches = dict(camera=model.encoders['camera'], lidar=model.encoders['lidar'], fuser=model.fuser)
    report['gradients'] = {}
    for name, module in branches.items():
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads, 'Missing gradients: ' + name
        finite = torch.stack([torch.isfinite(g).all() for g in grads]).all().item()
        magnitude = torch.stack([g.detach().abs().sum() for g in grads]).sum().item()
        assert finite and magnitude > 0, 'Invalid or zero gradients: ' + name
        report['gradients'][name] = dict(tensors=len(grads), finite=finite, absolute_sum=magnitude)
    report.update(loss=float(loss.detach()), points_unchanged=True,
                  peak_cuda_memory_mib=torch.cuda.max_memory_allocated() / 1024**2)
    for handle in handles:
        handle.remove()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
