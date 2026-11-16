# Fast-WAM

Fine-tuning of [Fast-WAM](https://arxiv.org/abs/2603.16666) (Wan2.2-TI2V-5B video expert + action expert,
`lerobot/fastwam_base`) on UniVTAC demonstrations, with tactile sensing as a third expert of the
Mixture-of-Transformers.

This repository covers training (cluster scripts) and the policy code.

## Model

One joint self-attention per layer over three experts:

| Expert | Weights | Tokens |
|---|---|---|
| video | Wan2.2-TI2V-5B | head \| wrist canvas, 224 x 448, Wan2.2-VAE latents |
| action | action expert of `lerobot/fastwam_base` (hidden 1024, 30 layers) | `action_horizon` actions, one per dataset frame |
| tactile | same architecture as the action expert, started from its weights | UniVTAC ResNet18 latents of the two marker streams: current frame clean, `tactile_future_steps` future frames `tactile_future_stride` dataset frames apart noised |

- Attention: video tokens see only video (`first_frame_causal`); action and tactile tokens see the current
  video frame and each other (`tactile_action_attention`).
- Action and touch each have their own flow time and loss; the tactile loss covers the future frames inside the
  episode.
- `freeze_video_expert=true`: the video is the current frame only. Under the Fast-WAM mask no trainable weight
  sees the future frames then, so they would only cost compute (`train_video_future`).
- Tactile encoder trained with the future-frame targets taken from its EMA copy (`tactile_encoder_trainable`),
  or frozen.
- Inference: the video is prefilled once into a K/V cache; actions and future touch are denoised jointly
  against it.

## Code

`src/lerobot` is a trimmed copy of [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0, see
`LICENSE`) at commit `fe1c545`, keeping the training stack and the `fastwam` policy. Changes against it:

- `policies/fastwam`: the tactile expert (`wan/tactile.py`, the UniVTAC encoder in `tactile_encoder.py`,
  `tactile_*` options), `camera_keys`, the current-frame video when the video expert is frozen, a prompt cache
  for the CPU text encoder.
- `datasets/factory.py`: per image key frame offsets (`image_delta_indices_for_key`), so the tactile streams
  load their own frames and the other images only the current one.
- Removed: the other policies, robots, teleoperators, motors, cameras, simulation environments and in-training
  environment evaluation, and the CLI tools other than `lerobot-train`.

## Setup (NHR alex)

```bash
git clone <this repo> ~/Projects/fast-wam
bash ~/Projects/fast-wam/scripts/install_cluster.sbatch   # venv + weights (~26 GB)
cp <UniVTAC>/encoder/checkpoints/resnet18/20251128-125750/best.pth \
    $WS/fastwam/ckpt/univtac_resnet18_20251128-125750.pth
```

Everything heavy lives on the anvme workspace (`$WS/fastwam`: venv, checkpoints, outputs; weights in `$WS/hf`).
The datasets are the UniVTAC LeRobot datasets (`$WS/smolvla/data/local/<task>-clean-100_vitac`).

## Train

```bash
sbatch scripts/train_cluster.sbatch lift_bottle
sbatch scripts/train_cluster.sbatch put_bottle_in_shelf
```

Environment variables: `VIDEO=frozen|train`, `TACTILE=none|frozen|trainable` (default trainable),
`TACTILE_STRIDE`, `TACTILE_STEPS`, `ACTION_HORIZON`, `BATCH_SIZE`, `STEPS`, `SAVE_FREQ`, `GRAD_CKPT`. `RESUME=1`
continues a run from its last checkpoint (`--dependency=afterany:<job>`). Runs log to Trackio, mirrored to wandb
by `scripts/sync_wandb.py`.

Measured on one A40 (batch 16, video expert frozen, trainable tactile encoder): 46 GB, about 1.9 s/step; 30000
steps take about 16 h.

## Test

CPU test with a tiny DiT (real Wan VAE and UniVTAC encoder): gradients per expert, tactile init from the action
expert, attention mask, joint sampling.

```bash
TACTILE_CKPT=$WS/fastwam/ckpt/univtac_resnet18_20251128-125750.pth HF_HOME=$WS/hf \
    [TRAINABLE=1] [FROZEN_VIDEO=0] $WS/fastwam/venv-wam/bin/python tests/test_tactile_expert_tiny.py
```
