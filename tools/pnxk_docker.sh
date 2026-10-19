#!/usr/bin/env bash
# Reuse the local RTX 5060 image with the current checkout, not its baked-in code.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_image="${PNXK_IMAGE:-bevfusion:rtx5060}"
task_action="${1:-check}"
if (($#)); then shift; fi
task_user="$(id -u):$(id -g)"

case "$task_action" in
  prepare)
    # Reuses binaries compiled for this image's Python/Torch/CUDA ABI. These
    # gitignored files must be refreshed if the runtime image changes.
    docker run --rm --user "$task_user" --entrypoint python \
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
  check|train)
    task_docker=(docker run --rm --gpus all --user "$task_user" --shm-size 2g
      --entrypoint python -v "$task_root:/work" -w /work
      -e PYTHONPATH=/work -e MPLCONFIGDIR=/tmp/pnxk-mpl "$task_image")
    if [[ "$task_action" == check ]]; then
      "${task_docker[@]}" tools/check_pnxk.py --root data/pnxk \
        --pipeline-config configs/pnxk/lidar.yaml "$@"
    else
      "${task_docker[@]}" tools/train.py configs/pnxk/lidar.yaml --launcher none \
        --run-dir runs/pnxk-lidar-single-gpu --max_epochs 1 "$@"
    fi
    ;;
  *)
    echo 'Usage: bash tools/pnxk_docker.sh {prepare|check|train} [additional arguments]' >&2
    exit 2
    ;;
esac
