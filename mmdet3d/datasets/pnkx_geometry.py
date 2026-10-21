"""Convert PNKX physical boxes to this repository's legacy box convention."""
import numpy as np


def raw_boxes_to_internal(boxes):
    """Center XYZ, LWH, CCW yaw -> center XYZ, WLH, legacy clockwise yaw.

    Follows the same convention as nuscenes_converter's ``-yaw - pi/2``.
    Z remains the geometric center; LiDARInstance3DBoxes handles bottom origin.
    """
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 7).copy()
    result[:, [3, 4]] = result[:, [4, 3]]
    result[:, 6] = -result[:, 6] - np.pi / 2
    return result


def internal_boxes_to_raw(boxes):
    """Inverse conversion for geometric-center boxes, not bottom-center tensors."""
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 7).copy()
    result[:, [3, 4]] = result[:, [4, 3]]
    result[:, 6] = -result[:, 6] - np.pi / 2
    result[:, 6] = (result[:, 6] + np.pi) % (2 * np.pi) - np.pi
    return result
