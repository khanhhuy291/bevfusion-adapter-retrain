"""Narrow compatibility fixes for the legacy MMCV runner on modern PyTorch."""
import torch


def patch_mmcv_scatter():
    # MMCV 1.x passes integer CUDA ids to a private Torch helper. New Torch
    # expects torch.device instead. Keep the rest of MMCV's scatter unchanged.
    import mmcv.parallel._functions as functions
    from mmcv.utils import digit_version

    if digit_version(torch.__version__) < digit_version('2.1.0'):
        return
    original = functions._get_stream
    if getattr(original, '_bevfusion_device_compat', False):
        return

    def get_stream(device):
        if isinstance(device, int):
            device = torch.device('cpu') if device == -1 else torch.device('cuda', device)
        return original(device)

    get_stream._bevfusion_device_compat = True
    functions._get_stream = get_stream
