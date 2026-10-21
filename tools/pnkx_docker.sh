#!/usr/bin/env bash
# Reuse the local RTX 5060 image with the current checkout, not its baked-in code.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_image="${PNKX_IMAGE:-bevfusion:rtx5060}"
task_action="${1:-check}"
if (($#)); then shift; fi
task_user="$(id -u):$(id -g)"

case "$task_action" in
  visualize-fusion)
    exec docker run --rm --gpus all --user "$task_user" --shm-size 2g \
      --entrypoint python -v "$task_root:/work" -w /work \
      -e PYTHONPATH=/work -e MPLCONFIGDIR=/tmp/pnkx-mpl "$task_image" \
      tools/visualize_pnkx.py "$@"
    ;;
  prepare)
    # Reuses binaries compiled for this image's Python/Torch/CUDA ABI. These
    # gitignored files must be refreshed if the runtime image changes.
    exec docker run --rm --user "$task_user" --entrypoint python \
      -v "$task_root:/host" "$task_image" -c '
from pathlib import Path
import shutil
source = Path("/workspace/bevfusion-adapter")
target = Path("/host")
files = list((source / "mmdet3d/ops").rglob("*.so"))
if not files:
    raise RuntimeError("Image does not contain the expected compiled extensions")
for file in files:
    destination = target / file.relative_to(source)
    if not destination.parent.is_dir():
        raise RuntimeError("Extension has no matching source directory: " + str(file))
    shutil.copy2(file, destination)
print("Copied", len(files), "CUDA extensions")
'
    ;;
  check|train|check-fusion|train-fusion|audit-fusion|check-debug|train-debug|check-fusion-debug|train-fusion-debug|audit-fusion-debug|check-reference|train-reference|check-fusion-reference|train-fusion-reference|audit-fusion-reference)
    task_mode=lidar
    if [[ "$task_action" == *fusion* ]]; then
      task_mode=fusion
    fi
    task_config="configs/pnkx/nuscenes/${task_mode}_8gb.yaml"
    task_run="runs/pnkx-${task_mode}-nuscenes-8gb"
    task_extra=()
    if [[ "$task_action" == *debug ]]; then
      task_config="configs/pnkx/${task_mode}.yaml"
      task_run="runs/pnkx-${task_mode}-debug"
      task_extra=(--max_epochs 1)
    elif [[ "$task_action" == *reference ]]; then
      task_config="configs/pnkx/nuscenes/${task_mode}.yaml"
      task_run="runs/pnkx-${task_mode}-nuscenes-reference"
    fi
    task_docker=(docker run --rm --gpus all --user "$task_user" --shm-size 2g
      --entrypoint python -v "$task_root:/work" -w /work
      -e PYTHONPATH=/work -e MPLCONFIGDIR=/tmp/pnkx-mpl "$task_image")
    if [[ "$task_action" == audit-fusion* ]]; then
      exec "${task_docker[@]}" tools/check_pnkx_fusion.py --config "$task_config" "$@"
    elif [[ "$task_action" == check* ]]; then
      exec "${task_docker[@]}" tools/check_pnkx.py --root data/pnkx \
        --pipeline-config "$task_config" "$@"
    else
      exec "${task_docker[@]}" tools/train.py "$task_config" --launcher none \
        --run-dir "$task_run" "${task_extra[@]}" "$@"
    fi
    ;;
  *)
    echo 'Usage: bash tools/pnkx_docker.sh {prepare|check|train|check-fusion|audit-fusion|train-fusion|visualize-fusion} [arguments]' >&2
    echo 'Default: nuScenes-style 8GB profile. Append -debug or -reference to an action to select another profile.' >&2
    exit 2
    ;;
esac
