"""Single-GPU PNKX inference with GT/pred overlays in ego BEV and cached images."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASS_NAMES = ('car', 'rider', 'pedestrian')
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(box):
    """Physical geometric-center LWH boxes, CCW yaw from ego +x."""
    x, y, z, length, width, height, yaw = box
    footprint = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length/2, width/2]
    c, s = np.cos(yaw), np.sin(yaw)
    footprint = footprint @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.column_stack((np.tile(footprint, (2, 1)), np.repeat([z-height/2, z+height/2], 4)))


def clip_segment_near(a, b, near=0.1):
    a, b = np.array(a, dtype=float, copy=True), np.array(b, dtype=float, copy=True)
    if a[2] < near and b[2] < near:
        return None
    if a[2] < near:
        a += (b-a) * ((near-a[2]) / (b[2]-a[2]))
    elif b[2] < near:
        b += (a-b) * ((near-b[2]) / (a[2]-b[2]))
    return a, b


def center_matches(gt, pred, distance=2.):
    """Greedy class-aware XY matches; heading error is diagnostic only."""
    used, matches = set(), []
    for p in sorted(pred, key=lambda item: -item['score']):
        candidates = [g for g in gt if g['class'] == p['class'] and g['id'] not in used]
        if not candidates:
            continue
        g = min(candidates, key=lambda g: np.linalg.norm(np.array(g['box'][:2])-p['box'][:2]))
        error = float(np.linalg.norm(np.array(g['box'][:2])-p['box'][:2]))
        if error > distance:
            continue
        yaw_error = abs((p['box'][6]-g['box'][6]+np.pi) % (2*np.pi)-np.pi)
        matches.append(dict(gt_id=g['id'], pred_id=p['id'], center_error_m=error,
                            yaw_error_deg=float(np.degrees(yaw_error))))
        used.add(g['id'])
    return matches


def draw_bev(path, points, gt, pred, bounds, zoom_radius, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Display +x upward, +y to the left: plot horizontal=-y, vertical=x.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    focus = min(gt, key=lambda g: np.linalg.norm(g['box'][:2])) if gt else (pred[0] if pred else None)
    cx, cy = focus['box'][:2] if focus else (0., 0.)
    limits = [(-bounds[4], -bounds[1], bounds[0], bounds[3]),
              (-cy-zoom_radius, -cy+zoom_radius, cx-zoom_radius, cx+zoom_radius)]
    for panel, (ax, limit) in enumerate(zip(axes, limits)):
        u, v = -points[:, 1], points[:, 0]
        mask = (u >= limit[0]) & (u <= limit[1]) & (v >= limit[2]) & (v <= limit[3])
        ax.scatter(u[mask][::2], v[mask][::2], s=.25, c='#adb5bd', rasterized=True)
        for items, color in ((gt, '#008f39'), (pred, '#e23d28')):
            for item in items:
                box = item['box']
                if not (limit[0] <= -box[1] <= limit[1] and limit[2] <= box[0] <= limit[3]):
                    continue
                corners = box_corners(box)[:4]
                loop = corners[[0, 1, 2, 3, 0]]
                ax.plot(-loop[:, 1], loop[:, 0], color=color, linewidth=1.1)
                length = max(.6, min(box[3]/2, 3.))
                ax.arrow(-box[1], box[0], -np.sin(box[6])*length, np.cos(box[6])*length,
                         color=color, head_width=.22, length_includes_head=True, linewidth=.8)
                if panel:
                    text = item['id'] + (' {:.2f}'.format(item['score']) if 'score' in item else '')
                    text += ' {:.0f}deg'.format(np.degrees(box[6]))
                    ax.text(-box[1], box[0], text, fontsize=6, color=color)
        ax.plot(0, 0, 'b^', markersize=7)
        ax.set(xlim=limit[:2], ylim=limit[2:], xlabel='-y ego [m] (left of vehicle is left)',
               ylabel='x ego [m] (forward is up)', aspect='equal')
        ax.grid(alpha=.2)
        ax.set_title('Full BEV' if panel == 0 else 'Zoom around nearest selected GT object')
    axes[0].legend(handles=[Line2D([0], [0], color='#008f39', label='GT'),
                            Line2D([0], [0], color='#e23d28', label='Prediction')])
    fig.suptitle(title + '\nArrows: annotated/predicted heading; IDs and yaw in zoom panel')
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def draw_cameras(path, root, cameras, gt, pred, width):
    import cv2
    panels = []
    for camera in cameras:
        image = cv2.imread(str(root / camera['data_path']))
        if image is None:
            raise ValueError('Cannot read cached image: ' + camera['data_path'])
        transform = np.asarray(camera['ego2camera'])
        intrinsic = np.asarray(camera['intrinsic'])
        def project(xyz):
            projected = xyz @ intrinsic.T
            return projected[:, :2] / projected[:, 2:3]
        def draw_segment(a, b, color, arrow=False):
            clipped = clip_segment_near(a, b)
            if clipped is None:
                return
            uv = np.clip(project(np.array(clipped)), -1000000, 1000000).round().astype(int)
            ok, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                         tuple(uv[0]), tuple(uv[1]))
            if ok:
                if arrow:
                    cv2.arrowedLine(image, start, end, color, 3, cv2.LINE_AA, tipLength=.2)
                else:
                    cv2.line(image, start, end, color, 2, cv2.LINE_AA)
        for items, color in ((gt, (57, 143, 0)), (pred, (40, 61, 226))):
            for item in items:
                box = item['box']
                corners = box_corners(box) @ transform[:3, :3].T + transform[:3, 3]
                for a, b in EDGES:
                    draw_segment(corners[a], corners[b], color)
                center = np.asarray(box[:3])
                front = center + [np.cos(box[6])*box[3]/2, np.sin(box[6])*box[3]/2, 0]
                heading = np.array([center, front]) @ transform[:3, :3].T + transform[:3, 3]
                draw_segment(*heading, color, arrow=True)
                if heading[0, 2] > .1:
                    u, v = project(heading[:1])[0]
                    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                        label = item['id'] + (' {:.2f}'.format(item['score']) if 'score' in item else '')
                        cv2.putText(image, label, (int(u), int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                                    .65, color, 2, cv2.LINE_AA)
        height = round(image.shape[0]*width/image.shape[1])
        image = cv2.resize(image, (width, height))
        name = camera['name']
        source = camera.get('source_name', name)
        cv2.rectangle(image, (0, 0), (width, 42), (20, 20, 20), -1)
        cv2.putText(image, name + (' <- '+source if source != name else '') + ' | GT green / Pred red',
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    if len(panels) != 6:
        raise ValueError('Expected six camera views')
    montage = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    if not cv2.imwrite(str(path), montage):
        raise IOError('Cannot write ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='runs/pnkx-fusion-full/configs.yaml')
    parser.add_argument('--checkpoint', default='runs/pnkx-fusion-full/epoch_6.pth')
    parser.add_argument('--dataset-root', help='Override the validation cache root')
    parser.add_argument('--out-dir', default='outputs/pnkx-val-predictions')
    parser.add_argument('--frames', nargs='+', type=int, help='Zero-based validation indices')
    parser.add_argument('--num-frames', type=int, default=5, help='Uniformly spaced indices if --frames is omitted')
    parser.add_argument('--classes', nargs='+', choices=CLASS_NAMES, default=['pedestrian'])
    parser.add_argument('--score-threshold', type=float, default=.1)
    parser.add_argument('--zoom-radius', type=float, default=10.)
    parser.add_argument('--camera-width', type=int, default=960)
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1 or args.num_frames < 1 or args.zoom_radius <= 0 or args.camera_width < 160:
        parser.error('Invalid score threshold, frame count, zoom radius or camera width')

    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import wrap_fp16_model
    from torchpack.utils.config import Config as TorchpackConfig
    from mmdet3d.datasets import build_dataset
    from mmdet3d.datasets.pnkx_geometry import internal_boxes_to_raw
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    from mmdet3d.utils.torch_compat import patch_mmcv_scatter

    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(0)
    patch_mmcv_scatter()
    settings = TorchpackConfig()
    settings.load(args.config, recursive=True)
    cfg = Config(recursive_eval(dict(settings)))
    val_cfg = cfg.data.val.copy()
    if args.dataset_root:
        val_cfg.dataset_root = str(Path(args.dataset_root).resolve()) + '/'
        val_cfg.ann_file = str(Path(val_cfg.dataset_root)/Path(val_cfg.ann_file).name)
    val_cfg.test_mode, val_cfg.filter_empty_gt = True, False
    dataset = build_dataset(val_cfg)
    indices = list(dict.fromkeys(args.frames)) if args.frames else np.linspace(
        0, len(dataset)-1, min(args.num_frames, len(dataset)), dtype=int).tolist()
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        parser.error('Frame indices must be between 0 and {}'.format(len(dataset)-1))
    # Load the complete trained state strictly; do not download/init pretrained.
    if cfg.model.encoders.get('camera'):
        cfg.model.encoders.camera.backbone.init_cfg = None
    model = build_model(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    if all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    print('Loaded full trained checkpoint:', args.checkpoint,
          '| epoch:', checkpoint.get('meta', {}).get('epoch'), '| evaluation only', flush=True)
    del checkpoint, state
    model = model.cuda().eval()
    use_fp16 = cfg.get('fp16') is not None
    if use_fp16:
        wrap_fp16_model(model)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    bounds = np.asarray(cfg.point_cloud_range)
    summary = dict(config=args.config, checkpoint=args.checkpoint, classes=args.classes,
                   score_threshold=args.score_threshold, box_convention='ego geometric center XYZ, LWH, CCW yaw radians',
                   precision='fp16' if use_fp16 else 'fp32', frames=[])
    for index in indices:
        info = dataset.data_infos[index]
        sample = dataset[index]
        np.testing.assert_allclose(sample['lidar_aug_matrix'].data.numpy(), np.eye(4), atol=1e-6,
                                   err_msg='Visualization requires identity validation LiDAR augmentation')
        points = sample['points'].data.numpy()
        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_fp16):
            prediction = model(**batch)[0]
        boxes = prediction['boxes_3d']
        geometric = torch.cat((boxes.gravity_center, boxes.tensor[:, 3:7]), dim=1).cpu().numpy()
        raw_predictions = internal_boxes_to_raw(geometric)
        scores = prediction['scores_3d'].cpu().numpy()
        labels = prediction['labels_3d'].cpu().numpy()
        if not np.isfinite(raw_predictions).all() or not np.isfinite(scores).all():
            raise ValueError('Non-finite prediction on frame {}'.format(index))
        gt = [dict(id='g'+str(i), **{'class': str(name)}, box=box.tolist())
              for i, (box, name) in enumerate(zip(info['gt_boxes'], info['gt_names']))]
        pred = [dict(id='p'+str(i), **{'class': CLASS_NAMES[int(label)]}, box=box.tolist(), score=float(score))
                for i, (box, score, label) in enumerate(zip(raw_predictions, scores, labels))]
        def selected(item):
            xyz = np.asarray(item['box'][:3])
            return (item['class'] in args.classes and item.get('score', 1.) >= args.score_threshold
                    and (xyz >= bounds[:3]).all() and (xyz < bounds[3:]).all())
        drawn_gt, drawn_pred = [g for g in gt if selected(g)], [p for p in pred if selected(p)]
        matches = center_matches(drawn_gt, drawn_pred)
        directory = output / '{:04d}_{}'.format(index, info['token'].split('/')[-1])
        directory.mkdir(parents=True, exist_ok=True)
        draw_bev(directory/'bev.png', points, drawn_gt, drawn_pred, bounds, args.zoom_radius,
                 'Val {} | {} | score >= {}'.format(index, ', '.join(args.classes), args.score_threshold))
        draw_cameras(directory/'cameras.jpg', Path(dataset.dataset_root), info['cams'],
                     drawn_gt, drawn_pred, args.camera_width)
        matched_gt, matched_pred = {m['gt_id'] for m in matches}, {m['pred_id'] for m in matches}
        details = dict(index=index, token=info['token'], ground_truth=gt, predictions=pred,
                       drawn_gt_ids=[g['id'] for g in drawn_gt], drawn_pred_ids=[p['id'] for p in drawn_pred],
                       matches_at_2m=matches, unmatched_gt_ids=[g['id'] for g in drawn_gt if g['id'] not in matched_gt],
                       unmatched_pred_ids=[p['id'] for p in drawn_pred if p['id'] not in matched_pred])
        (directory/'boxes.json').write_text(json.dumps(details, indent=2))
        row = dict(index=index, token=info['token'], drawn_gt=len(drawn_gt), drawn_predictions=len(drawn_pred),
                   matched_at_2m=len(matches), directory=str(directory))
        summary['frames'].append(row)
        print(json.dumps(row), flush=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved BEV, six-camera overlays and raw-convention boxes to', output)


if __name__ == '__main__':
    main()
