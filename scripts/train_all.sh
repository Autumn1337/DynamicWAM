#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_ROOT/.." && pwd)
ENV_ROOT=${DYNAMICWAM_ENV_ROOT:-$ROOT/.venv}
PYTHON=$ENV_ROOT/bin/python
TORCHRUN=$ENV_ROOT/bin/torchrun
ACCELERATE=$ENV_ROOT/bin/accelerate
PROFILE=$ROOT/configs/absolute_motion_v2.yaml
RUN_ROOT=${DYNAMICWAM_RUN_ROOT:-$ROOT/outputs/mainline_dynamic_only}
DATASET=$ROOT/data/packed/domino_absolute_motion_v2/dataset.json
PCA=$ROOT/outputs/stage1_pca/pca_stats.pt
STAGE1=$ROOT/outputs/training/absolute_motion_stage1_video/exports/stage1_step_80000.pt
INITIALIZED=$ROOT/outputs/checkpoint_init/absolute_motion_init.pt
STAGE2=$ROOT/outputs/training/absolute_motion_stage2_action/exports/stage2_step_80000.pt
STAGE3=$ROOT/outputs/training/absolute_motion_stage3_joint/exports/stage3_step_40000.pt
GPU_PROCESSES=${DYNAMICWAM_GPU_PROCESSES:-8}

export PATH=$ENV_ROOT/bin:$PATH
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

for executable in "$PYTHON" "$TORCHRUN" "$ACCELERATE"; do
  if [[ ! -x $executable ]]; then
    printf 'missing environment executable: %s\n' "$executable" >&2
    exit 1
  fi
done

if [[ ! -s $DATASET ]]; then
  printf 'missing dynamic dataset: %s\n' "$DATASET" >&2
  exit 1
fi

printf '%s dynamic_only_mainline_start dataset=%s gpus=%s\n' \
  "$(date '+%F %T')" "$DATASET" "$CUDA_VISIBLE_DEVICES"

if [[ ! -s $PCA ]]; then
  "$TORCHRUN" --standalone --nproc_per_node="$GPU_PROCESSES" \
    "$ROOT/scripts/train.py" stage1_pca --config "$PROFILE" \
    2>&1 | tee "$RUN_ROOT/stage1_pca.log"
fi

if [[ ! -s $STAGE1 ]]; then
  "$ACCELERATE" launch --multi_gpu --num_processes "$GPU_PROCESSES" \
    --main_process_port 29521 \
    "$ROOT/scripts/train.py" stage1 --config "$PROFILE" \
    2>&1 | tee "$RUN_ROOT/stage1.log"
fi

if [[ ! -s $INITIALIZED ]]; then
  "$PYTHON" "$ROOT/scripts/initialize_motion_checkpoint.py" \
    --config "$PROFILE" \
    2>&1 | tee "$RUN_ROOT/initialize.log"
fi

if [[ ! -s $STAGE2 ]]; then
  "$ACCELERATE" launch --multi_gpu --num_processes "$GPU_PROCESSES" \
    --main_process_port 29522 \
    "$ROOT/scripts/train.py" stage2 --config "$PROFILE" \
    2>&1 | tee "$RUN_ROOT/stage2.log"
fi

if [[ ! -s $STAGE3 ]]; then
  "$ACCELERATE" launch --multi_gpu --num_processes "$GPU_PROCESSES" \
    --main_process_port 29523 \
    "$ROOT/scripts/train.py" stage3 --config "$PROFILE" \
    2>&1 | tee "$RUN_ROOT/stage3.log"
fi

printf '%s dynamic_only_mainline_complete checkpoint=%s\n' \
  "$(date '+%F %T')" "$STAGE3"
