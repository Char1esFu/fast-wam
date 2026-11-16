#!/bin/bash -l
# Build the Fast-WAM environment on the anvme workspace: upstream lerobot pinned to LEROBOT_REV with
# patches/*.patch (branch univtac), its own .venv, and the weights in $HF_HOME. Runs on a login node too
# (bash scripts/install_cluster.sbatch): it needs no GPU, only the proxy.
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

REPO=${REPO:-$HOME/Projects/fastwam-univtac}
WS=${WS:-$(ws_find univtac)}
ROOT=${FASTWAM_ROOT:-$WS/fastwam}
LEROBOT_REV=${LEROBOT_REV:-fe1c545}  # huggingface/lerobot main, 2026-11-13
export HF_HOME=${HF_HOME:-$WS/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WS/uv-cache} UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1  # test artifacts only

mkdir -p "$ROOT/ckpt" "$ROOT/outputs"
cd "$ROOT"
if [ ! -d lerobot/.git ]; then
    git clone -q --filter=blob:none https://github.com/huggingface/lerobot.git lerobot
fi
git -C lerobot fetch -q origin
git -C lerobot switch -q --detach "$LEROBOT_REV"
git -C lerobot switch -q -C univtac
git -C lerobot -c user.name=univtac -c user.email=univtac@localhost am -q "$REPO"/patches/*.patch
echo "[install] lerobot $(git -C lerobot log --oneline "$LEROBOT_REV"..HEAD | tr '\n' ';') on $LEROBOT_REV"

[ -x .venv/bin/python ] || uv venv -q .venv --python 3.12
VIRTUAL_ENV=$PWD/.venv uv pip install -q -e "./lerobot[fastwam,training]" --torch-backend=cu128

# Weights (~26 GB): FastWAM base (Wan2.2-5B video expert + action expert), the Wan2.2 VAE and UMT5 text
# encoder, the UMT5 tokenizer.
.venv/bin/hf download lerobot/fastwam_base > /dev/null
.venv/bin/hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers model_index.json vae/config.json \
    vae/diffusion_pytorch_model.safetensors text_encoder/config.json text_encoder/model.safetensors.index.json \
    text_encoder/model-00001-of-00003.safetensors text_encoder/model-00002-of-00003.safetensors \
    text_encoder/model-00003-of-00003.safetensors > /dev/null
.venv/bin/hf download google/umt5-xxl config.json generation_config.json special_tokens_map.json spiece.model \
    tokenizer.json tokenizer_config.json > /dev/null
[ -f ckpt/univtac_resnet18_20251128-125750.pth ] || \
    echo "[install] copy the UniVTAC tactile encoder checkpoint to $ROOT/ckpt/univtac_resnet18_20251128-125750.pth"
.venv/bin/python -c "
import torch, lerobot
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
print('torch', torch.__version__, 'lerobot', lerobot.__file__)"
echo "INSTALL DONE"
