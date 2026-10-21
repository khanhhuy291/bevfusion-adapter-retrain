"""Convert recording-level PNKX splits without importing the training stack."""
import argparse
import bisect
import datetime as dt
import json
import pickle
from pathlib import Path

import numpy as np

CAMERAS = ('CAM_P_F', 'CAM_P_FL', 'CAM_P_FR', 'CAM_P_B', 'CAM_P_LB', 'CAM_P_RB')
# Logical model slots stay fixed; VF6_03's rear sensor has a different raw name.
# This is an explicit vehicle setup, not a fallback between calibrations.
CAMERA_SOURCES_BY_VEHICLE = {'VF6_03': {'CAM_P_B': 'CAM_P_B_New'}}
CLASSES = ('car', 'rider', 'pedestrian')


def camera_sources(recording, cameras=CAMERAS):
    vehicle = '_'.join(recording.split('_')[2:4])
    overrides = CAMERA_SOURCES_BY_VEHICLE.get(vehicle, {})
    return {camera: overrides.get(camera, camera) for camera in cameras}


def portable_metadata(value):
    """Avoid NumPy-version-specific pickle globals in caches made on new Python."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [portable_metadata(item) for item in value]
    return value


def timestamp_ns(path):
    seconds, nanos = Path(path).stem.split('-')
    if not (seconds.isdigit() and nanos.isdigit() and len(nanos) == 9):
        raise ValueError('Expected seconds-nanoseconds with 9 fractional digits: {}'.format(path))
    return int(seconds) * 10**9 + int(nanos)


def read_split(path, root):
    """Accept recording names or historic paths ending in data/<recording>."""
    recordings = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        recording = Path(line.replace('\\', '/')).name
        if not (root / 'data' / recording).is_dir():
            raise FileNotFoundError('Recording missing under data root: {}'.format(recording))
        if recording in recordings:
            raise ValueError('Duplicate recording in {}: {}'.format(path, recording))
        recordings.append(recording)
    if not recordings:
        raise ValueError('Empty split: {}'.format(path))
    return recordings


def calibration(root, recording, sensors):
    parts = recording.split('_')
    vehicle = '_'.join(parts[2:4])
    date = dt.datetime.strptime(parts[0], '%Y%m%d').date().isoformat()
    dated = root / '{}_Extrinsics_By_Dates.json'.format(vehicle)
    if dated.exists():
        periods = json.loads(dated.read_text())['periods']
        matches = [p for p in periods if p['valid_from'] <= date < p['valid_to']]
        if len(matches) != 1:
            raise ValueError('Expected exactly one calibration period for {} ({})'.format(recording, date))
        extrinsics = matches[0]['extrinsics']
    else:
        extrinsics = json.loads((root / '{}_Extrinsics.json'.format(vehicle)).read_text())
    for sensor in sensors:
        if sensor not in extrinsics:
            raise ValueError('Missing calibration {} in {}'.format(sensor, recording))
        matrix = np.asarray(extrinsics[sensor], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Invalid extrinsics for {}'.format(sensor))
        if not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4):
            raise ValueError('Invalid rigid transform for {}'.format(sensor))
    intrinsics = {}
    camera_sensors = [s for s in sensors if s.startswith('CAM_')]
    if camera_sensors:
        intrinsics = json.loads((root / '{}_Intrinsics.json'.format(vehicle)).read_text())
        for sensor in camera_sensors:
            if sensor not in intrinsics:
                raise ValueError('Missing intrinsics {} in {}'.format(sensor, recording))
    return extrinsics, intrinsics


def read_labels(path):
    boxes, names, occlusion, counts, tracks = [], [], [], [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 11:
            raise ValueError('{}:{}: expected 11 fields'.format(path, number))
        box = np.asarray(values[:7], dtype=np.float32)
        name = values[7].lower()
        if name not in CLASSES or not np.isfinite(box).all() or (box[3:6] <= 0).any():
            raise ValueError('{}:{}: invalid box or class'.format(path, number))
        occ, count, track = map(int, values[8:])
        if count < 0:
            raise ValueError('Negative point count in {}'.format(path))
        boxes.append(box); names.append(name)
        occlusion.append(occ); counts.append(count); tracks.append(track)
    return dict(gt_boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
                gt_names=np.asarray(names), occlusion=np.asarray(occlusion),
                annotation_num_points=np.asarray(counts), track_ids=np.asarray(tracks))


def sensor_index(folder, suffix):
    files = sorted(folder.glob('*' + suffix), key=timestamp_ns)
    if not files:
        raise FileNotFoundError('No {} files in {}'.format(suffix, folder))
    return [timestamp_ns(f) for f in files], files


def nearest(index, timestamp, max_delta_ms):
    times, files = index
    i = bisect.bisect_left(times, timestamp)
    j = min((j for j in (i - 1, i) if 0 <= j < len(times)),
            key=lambda j: (abs(times[j] - timestamp), times[j]))
    delta = abs(times[j] - timestamp) / 1e6
    if delta > max_delta_ms:
        raise ValueError('Nearest sensor file {} is {:.3f} ms away (limit {})'.format(
            files[j], delta, max_delta_ms))
    return files[j], delta


def convert(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    cameras = CAMERAS if args.with_cameras else ()
    if cameras and (args.camera_extrinsics is None or args.distortion_model is None):
        raise ValueError('Camera conversion requires explicit --camera-extrinsics and --distortion-model')
    splits = {s: read_split(getattr(args, s + '_split'), root) for s in ('train', 'val')}
    overlap = sorted(set(splits['train']) & set(splits['val']))
    if overlap and not args.allow_split_overlap:
        raise ValueError('Train/val recording overlap; debug only: --allow-split-overlap: {}'.format(overlap))
    import laspy

    output.mkdir(parents=True, exist_ok=True)
    report = dict(overlapping_recordings=overlap, debug_overlap=bool(overlap),
                  camera_extrinsics=args.camera_extrinsics,
                  distortion_model=args.distortion_model, camera_sources={}, splits={})
    memo = {}
    for split, recordings in splits.items():
        infos, skipped, issues = [], [], []
        for recording in recordings:
            sources = camera_sources(recording, cameras)
            report['camera_sources'][recording] = sources
            ext, intr = calibration(root, recording, ('LIDAR_TOP',) + tuple(sources.values()))
            for chunk in sorted((root / 'data' / recording).iterdir()):
                if not chunk.is_dir():
                    continue
                label_dir = root / 'label' / recording / chunk.name
                if not label_dir.is_dir() or not list(label_dir.glob('*.txt')):
                    message = 'No labels: {}/{}'.format(recording, chunk.name)
                    if not args.skip_missing_labels:
                        raise FileNotFoundError(message)
                    issues.append(message)
                    continue
                indexes = {'LIDAR_TOP': sensor_index(chunk / 'LIDAR' / 'LIDAR_TOP', '.laz')}
                indexes.update({c: sensor_index(chunk / 'CAMERA' / sources[c], '.jpg') for c in cameras})
                labels = sorted(label_dir.glob('*.txt'), key=timestamp_ns)
                if args.max_frames_per_chunk:
                    labels = labels[:args.max_frames_per_chunk]
                for label in labels:
                    token = '{}/{}/{}'.format(recording, chunk.name, label.stem)
                    if token in memo:
                        infos.append(memo[token]); continue
                    timestamp = timestamp_ns(label)
                    try:
                        lidar, lidar_delta = nearest(indexes['LIDAR_TOP'], timestamp, args.lidar_max_delta_ms)
                        matches = {c: nearest(indexes[c], timestamp, args.camera_max_delta_ms) for c in cameras}
                    except ValueError as exc:
                        if not args.skip_unmatched:
                            raise
                        skipped.append(dict(token=token, reason=str(exc))); continue
                    annotation = read_labels(label)
                    cache = output / 'points' / recording / chunk.name / (label.stem + '.bin')
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cloud = laspy.read(lidar)
                    xyz = np.column_stack((cloud.x, cloud.y, cloud.z))
                    t = np.asarray(ext['LIDAR_TOP'], dtype=np.float64)
                    xyz = xyz @ t[:3, :3].T + t[:3, 3]
                    # Single sweep: xyz in ego, raw intensity, zero sweep-time lag.
                    points = np.column_stack((xyz, cloud.intensity, np.zeros(len(xyz)))).astype(np.float32)
                    if not np.isfinite(points).all():
                        raise ValueError('Non-finite points: {}'.format(lidar))
                    points.tofile(cache)
                    cams = []
                    if cameras:
                        import cv2
                    for camera, (image_path, delta) in matches.items():
                        source = sources[camera]
                        params = intr[source]
                        k = np.asarray(params['camera_matrix'], dtype=np.float64)
                        distortion = np.asarray(params['distortion_coefficients'], dtype=np.float64)
                        image = cv2.imread(str(image_path))
                        if image is None:
                            raise ValueError('Cannot decode {}'.format(image_path))
                        if image.shape[:2] != (params['image_height'], params['image_width']):
                            raise ValueError('Image/calibration resolution mismatch: {}'.format(image_path))
                        image = cv2.undistort(image, k, distortion, None, k)
                        destination = output / 'images' / recording / chunk.name / camera / image_path.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(destination), image):
                            raise IOError('Cannot write {}'.format(destination))
                        t = np.asarray(ext[source], dtype=np.float32)
                        if args.camera_extrinsics == 'camera-to-ego':
                            t = np.linalg.inv(t)
                        cams.append(dict(name=camera, source_name=source,
                                         source_path=str(image_path.relative_to(root)),
                                         data_path=str(destination.relative_to(output)),
                                         ego2camera=t, intrinsic=k.astype(np.float32), delta_ms=delta))
                    info = dict(token=token, recording=recording, chunk=chunk.name,
                                timestamp=timestamp / 1e9, timestamp_ns=timestamp,
                                lidar_path=str(cache.relative_to(output)), lidar_delta_ms=lidar_delta,
                                num_points=len(points), cams=cams, **annotation)
                    memo[token] = info; infos.append(info)
        if not infos:
            raise ValueError('No usable samples in {}'.format(split))
        metadata = dict(schema_version=1, classes=CLASSES, point_frame='ego',
                        box_origin='geometric_center', point_features=['x', 'y', 'z', 'intensity', 'sweep_dt'],
                        recordings=recordings, split_overlap=overlap, cameras=list(cameras),
                        camera_sources={r: report['camera_sources'][r] for r in recordings})
        with (output / ('pnkx_infos_' + split + '.pkl')).open('wb') as handle:
            pickle.dump(portable_metadata(dict(metadata=metadata, infos=infos)), handle, protocol=4)
        report['splits'][split] = dict(samples=len(infos), recordings=recordings,
                                      skipped_samples=skipped, skipped_chunks=issues)
        print('{}: {} samples, {} skipped frames, {} skipped chunks'.format(
            split, len(infos), len(skipped), len(issues)), flush=True)
    (output / 'conversion_report.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='Data-PNKX')
    parser.add_argument('--output', default='data/pnkx')
    parser.add_argument('--train-split', default='Data-PNKX/split/train.txt')
    parser.add_argument('--val-split', default='Data-PNKX/split/val.txt')
    parser.add_argument('--allow-split-overlap', action='store_true')
    parser.add_argument('--with-cameras', action='store_true')
    parser.add_argument('--camera-extrinsics', choices=['ego-to-camera', 'camera-to-ego'])
    parser.add_argument('--distortion-model', choices=['opencv-standard'])
    parser.add_argument('--lidar-max-delta-ms', type=float, default=1.0)
    parser.add_argument('--camera-max-delta-ms', type=float, default=40.0)
    parser.add_argument('--skip-unmatched', action='store_true')
    parser.add_argument('--skip-missing-labels', action='store_true')
    parser.add_argument('--max-frames-per-chunk', type=int, default=0,
                        help='Debug: keep only first N labelled frames; 0 means all')
    args = parser.parse_args()
    if min(args.lidar_max_delta_ms, args.camera_max_delta_ms, args.max_frames_per_chunk) < 0:
        parser.error('Timestamp limits and frame count must be nonnegative')
    convert(args)


if __name__ == '__main__':
    main()
