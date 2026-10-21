"""CPU diagnostic AP using greedy, class-aware XY center matching.

This metric ignores box size/yaw and is neither nuScenes NDS nor IoU AP.
"""
import numpy as np


def center_distance_ap(ground_truth, predictions, classes, thresholds=(0.5, 1., 2., 4.)):
    metrics = {}
    for threshold in thresholds:
        aps = []
        for class_id, name in enumerate(classes):
            targets = [g['centers'][g['labels'] == class_id, :2] for g in ground_truth]
            total = sum(len(t) for t in targets)
            proposals = []
            for frame, pred in enumerate(predictions):
                for i in np.flatnonzero(pred['labels'] == class_id):
                    proposals.append((float(pred['scores'][i]), frame, pred['centers'][i, :2]))
            proposals.sort(key=lambda p: -p[0])
            matched = [np.zeros(len(t), dtype=bool) for t in targets]
            tp = np.zeros(len(proposals)); fp = np.ones(len(proposals))
            for i, (_, frame, center) in enumerate(proposals):
                distances = np.linalg.norm(targets[frame] - center, axis=1)
                distances[matched[frame]] = np.inf
                if len(distances) and distances.min() <= threshold:
                    matched[frame][distances.argmin()] = True
                    tp[i], fp[i] = 1, 0
            recall = np.cumsum(tp) / max(total, 1)
            precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1)
            # All-points interpolated area under precision/recall curve.
            r = np.r_[0., recall, 1.]
            pr = np.r_[0., precision, 0.]
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            ap = float(np.sum((r[1:] - r[:-1]) * pr[1:])) if total else 0.
            metrics['center_AP/{}@{}m'.format(name, threshold)] = ap
            metrics['center_GT/{}'.format(name)] = total
            if total:
                aps.append(ap)
        metrics['center_mAP@{}m'.format(threshold)] = float(np.mean(aps)) if aps else 0.
    return metrics
