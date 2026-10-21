"""Resolve inherited configs and export PNKX/nuScenes comparison evidence."""
import argparse
import importlib.util
import json
from pathlib import Path

from torchpack.utils.config import Config


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    config = Config()
    config.load(str(ROOT / path), recursive=True)
    spec = importlib.util.spec_from_file_location('bevfusion_config_eval', ROOT / 'mmdet3d/utils/config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recursive_eval(dict(config))


def differences(left, right, path='model'):
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows.append(dict(path=path + '.' + key, pnkx=left.get(key),
                                 nuscenes=right.get(key),
                                 missing_in='pnkx' if key not in left else 'nuscenes'))
            else:
                rows.extend(differences(left[key], right[key], path + '.' + key))
        return rows
    if left != right:
        return [dict(path=path, pnkx=left, nuscenes=right)]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/pnkx-config-review')
    args = parser.parse_args()
    paths = dict(
        pnkx_fusion='configs/pnkx/fusion.yaml',
        pnkx_lidar='configs/pnkx/lidar.yaml',
        pnkx_reference_fusion='configs/pnkx/nuscenes/fusion.yaml',
        pnkx_reference_lidar='configs/pnkx/nuscenes/lidar.yaml',
        pnkx_8gb_fusion='configs/pnkx/nuscenes/fusion_8gb.yaml',
        pnkx_8gb_lidar='configs/pnkx/nuscenes/lidar_8gb.yaml',
        nuscenes_fusion='configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml',
        nuscenes_lidar='configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml')
    resolved = {name: resolve(path) for name, path in paths.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved-configs.json').write_text(json.dumps(resolved, indent=2))
    for mode in ('fusion', 'lidar'):
        rows = differences(resolved['pnkx_' + mode]['model'], resolved['nuscenes_' + mode]['model'])
        (output / (mode + '-model-differences.json')).write_text(json.dumps(rows, indent=2))
        print('{}: {} differing model fields'.format(mode, len(rows)))
        for profile in ('reference', '8gb'):
            rows = differences(resolved['pnkx_' + profile + '_' + mode]['model'],
                               resolved['nuscenes_' + mode]['model'])
            (output / (profile + '-' + mode + '-model-differences.json')).write_text(json.dumps(rows, indent=2))
            print('{} {}: {} differing model fields'.format(profile, mode, len(rows)))
    (output / 'sources.json').write_text(json.dumps(paths, indent=2))
    print('Wrote resolved config evidence to {}'.format(output))


if __name__ == '__main__':
    main()
