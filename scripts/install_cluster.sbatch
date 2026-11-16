#!/bin/bash -l
# Build the training environment on the anvme workspace: a venv with this repo installed (editable) and the
# weights in $HF_HOME (~26 GB). Needs no GPU, only the proxy: `bash scripts/install_cluster.sbatch` on a login node
# works too.
#   sbatch scripts/install_cluster.sbatch
#SBATCH --job-name=fastwam-install
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=fastwam-install-%j.out
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-base
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export PATH=$HOME/.local/bin:$PATH

REPO=${REPO:-$HOME/Projects/fast-wam}
WS=${WS:-$(ws_find univtac)}
ROOT=${FASTWAM_ROOT:-$WS/fastwam}
VENV=${VENV:-$ROOT/venv-wam}
export HF_HOME=${HF_HOME:-$WS/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WS/uv-cache} UV_LINK_MODE=copy

mkdir -p "$ROOT/ckpt" "$ROOT/outputs" "$ROOT/logs"
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
VIRTUAL_ENV=$VENV uv pip install -q -e "$REPO" --torch-backend=cu128
(cd /tmp && "$VENV/bin/python" -c "
import torch, lerobot
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
print('torch', torch.__version__, 'lerobot package', lerobot.__file__)")

# Weights: FastWAM base (Wan2.2-5B video expert + action expert), the Wan2.2 VAE and UMT5 text encoder, the UMT5
# tokenizer.
"$VENV/bin/hf" download lerobot/fastwam_base > /dev/null
"$VENV/bin/hf" download Wan-AI/Wan2.2-TI2V-5B-Diffusers model_index.json vae/config.json \
    vae/diffusion_pytorch_model.safetensors text_encoder/config.json text_encoder/model.safetensors.index.json \
    text_encoder/model-00001-of-00003.safetensors text_encoder/model-00002-of-00003.safetensors \
    text_encoder/model-00003-of-00003.safetensors > /dev/null
"$VENV/bin/hf" download google/umt5-xxl config.json generation_config.json special_tokens_map.json spiece.model \
    tokenizer.json tokenizer_config.json > /dev/null
[ -f "$ROOT/ckpt/univtac_resnet18_20251128-125750.pth" ] || \
    echo "[install] copy the UniVTAC tactile encoder (encoder/checkpoints/resnet18/20251128-125750/best.pth)" \
         "to $ROOT/ckpt/univtac_resnet18_20251128-125750.pth"
echo "INSTALL DONE"
