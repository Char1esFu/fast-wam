#!/bin/bash -l
# Fine-tune Fast-WAM (`fastwam` of src/lerobot: Wan2.2-TI2V-5B video expert + action expert + tactile expert) on one
# NHR alex A40, on the UniVTAC LeRobot datasets (built by the UniVTAC SmolVLA pipeline).
#   sbatch scripts/train_cluster.sbatch put_bottle_in_shelf [trainer args]
# Needs scripts/install_cluster.sbatch first. Everything lives on the anvme workspace.
#
# MoT experts, one joint attention per layer:
#   video: Wan2.2-5B over the head | wrist canvas (224 x 448). VIDEO=frozen: frozen, and the video is the
#     current frame (with the video frozen the future frames can not change anything, see the config's
#     `train_video_future`). VIDEO=train: the expert trains on the current + future frames (Fast-WAM recipe,
#     needs ACTION_HORIZON divisible by 8).
#   action: ACTION_HORIZON actions, one per dataset frame (30 fps); attends to the current video frame.
#   tactile (TACTILE=none|frozen|trainable): UniVTAC ResNet18 latents of the two marker streams, the current
#     frame clean and TACTILE_STEPS future frames TACTILE_STRIDE dataset frames apart denoised with the
#     actions. Same architecture as the action expert (30 layers), started from its weights.
#     trainable (default): the encoder trains, the future targets come from its EMA.
#
# RESUME=1 continues the run in OUTPUT_DIR from its last checkpoint (24 h limit):
#   sbatch --dependency=afterany:<job> --export=NONE,RESUME=1,<same env> scripts/train_cluster.sbatch <task>
#SBATCH --job-name=fastwam-train
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=fastwam-train-%x-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base cuda/12.8.1
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

if [ $# -gt 0 ] && [[ $1 != --* ]]; then TASK_NAME=$1; shift; fi
TASK_NAME=${TASK_NAME:?usage: sbatch train_cluster.sbatch <task> [trainer args]}
TASK_CONFIG=${TASK_CONFIG:-clean}
EXPERT_DATA_NUM=${EXPERT_DATA_NUM:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
REPO=${REPO:-$HOME/Projects/fast-wam}
WS=${WS:-$(ws_find univtac)}
ROOT=${FASTWAM_ROOT:-$WS/fastwam}
VENV=${VENV:-$ROOT/venv-wam}  # built by scripts/install_cluster.sbatch
TACTILE_CKPT=${TACTILE_CKPT:-$ROOT/ckpt/univtac_resnet18_20251128-125750.pth}
# Weights and caches on NVMe ($WORK is NFS and can stall at a few MB/s).
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # weights are downloaded by install_cluster.sbatch
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$WS/smolvla/hf_datasets}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=${BATCH_SIZE:-16}
ACTION_HORIZON=${ACTION_HORIZON:-50}
VIDEO=${VIDEO:-frozen}
TACTILE=${TACTILE:-trainable}
TACTILE_STRIDE=${TACTILE_STRIDE:-5}
TACTILE_STEPS=${TACTILE_STEPS:-6}
GRAD_CKPT=${GRAD_CKPT:-false}
STEPS=${STEPS:-30000}
SAVE_FREQ=${SAVE_FREQ:-10000}
TRACKIO_PROJECT=${TRACKIO_PROJECT:-fastwam_univtac}  # Trackio database under $HF_HOME/trackio
WANDB_SYNC=${WANDB_SYNC:-1}  # background scripts/sync_wandb.py mirror of the Trackio run (needs ~/.netrc)
WANDB_SYNC_S=${WANDB_SYNC_S:-300}
export WANDB_ENTITY=${WANDB_ENTITY:-cong_fu}
export WANDB_PROJECT=${WANDB_PROJECT:-fastwam_univtac}

SETTINGS=${TASK_NAME}-${TASK_CONFIG}-${EXPERT_DATA_NUM}
REPO_ID=local/${SETTINGS}_vitac
DATASET_ROOT=${DATASET_ROOT:-$WS/smolvla/data/$REPO_ID}
[ -f "$DATASET_ROOT/meta/info.json" ] || {
    echo "missing $DATASET_ROOT: build it with UniVTAC policy/smolvla/scripts/train_cluster.sbatch $TASK_NAME"; exit 1; }

cd "$ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "[fastwam] task=$TASK_NAME dataset=$DATASET_ROOT" \
    "repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo uncommitted)"

RUN_TAG=${RUN_TAG:-${SETTINGS}_h${ACTION_HORIZON}_video-${VIDEO}_tac-${TACTILE}-s${TACTILE_STRIDE}x${TACTILE_STEPS}}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/fastwam_${RUN_TAG}}

if [ "$WANDB_SYNC" = 1 ]; then
    SYNC=("${UV:-$HOME/.local/bin/uv}" run --no-project --with wandb python "$REPO/scripts/sync_wandb.py"
        --project "$TRACKIO_PROJECT" --run-filter "fastwam_${RUN_TAG}-")
    "${SYNC[@]}" --loop "$WANDB_SYNC_S" &
    SYNC_PID=$!
    trap 'kill $SYNC_PID 2>/dev/null; "${SYNC[@]}" || true' EXIT
fi

if [ "${RESUME:-0}" = 1 ]; then
    LAST=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json
    [ -f "$LAST" ] || { echo "[resume] no checkpoint in $OUTPUT_DIR"; exit 1; }
    echo "[resume] $OUTPUT_DIR from $(cat "$OUTPUT_DIR/checkpoints/last/training_state/training_step.json" 2>/dev/null || echo '?')"
    "$VENV"/bin/lerobot-train --config_path="$LAST" --resume=true "$@"
    exit
fi

CFG=${TMPDIR:-/tmp}/fastwam_${RUN_TAG}.yml
{
    echo "policy:"
    echo "  type: fastwam"
    echo "  push_to_hub: false"
    echo "  action_dim: 9"
    echo "  proprio_dim: 16"
    echo "  action_horizon: $ACTION_HORIZON"
    echo "  n_action_steps: $ACTION_HORIZON"
    echo "  image_size: [224, 448]"
    echo "  camera_keys: [observation.images.head, observation.images.wrist]"
    echo "  freeze_video_expert: $([ "$VIDEO" = frozen ] && echo true || echo false)"
    echo "  use_gradient_checkpointing: $GRAD_CKPT"
    echo "  text_encoder_device: cpu"  # UMT5-XXL off the GPU; prompts are cached per distinct task string
    if [ "$TACTILE" != none ]; then
        echo "  tactile_keys: [observation.images.tactile_ll_marker, observation.images.tactile_lr_marker]"
        echo "  tactile_future_stride: $TACTILE_STRIDE"
        echo "  tactile_future_steps: $TACTILE_STEPS"
        echo "  tactile_encoder_path: $TACTILE_CKPT"
        echo "  tactile_encoder_trainable: $([ "$TACTILE" = trainable ] && echo true || echo false)"
    fi
} > "$CFG"
cat "$CFG"

"$VENV"/bin/lerobot-train \
    --config_path="$CFG" \
    --dataset.repo_id="$REPO_ID" --dataset.root="$DATASET_ROOT" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" \
    --output_dir="$OUTPUT_DIR" --job_name="fastwam_${RUN_TAG}" \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq=50 \
    --trackio.enable=true --trackio.project="$TRACKIO_PROJECT" "$@"
