# Fast-WAM + tactile expert on UniVTAC

Fine-tuning of [Fast-WAM](https://arxiv.org/abs/2603.16666) (lerobot `fastwam`, pinned upstream lerobot
`fe1c545`) on the UniVTAC simulation tasks, with tactile sensing as its own expert of the
Mixture-of-Transformers. Cluster: NHR alex (A40), everything heavy on the anvme workspace.

## Model

One joint self-attention per layer over three experts:

| Expert | Weights | Tokens |
|---|---|---|
| video | Wan2.2-TI2V-5B (from `lerobot/fastwam_base`) | head \| wrist canvas, 224 x 448, Wan2.2 VAE latents |
| action | action expert of `lerobot/fastwam_base` (hidden 1024, 30 layers) | `action_horizon` actions, 30 fps |
| tactile | new, same architecture as the action expert, started from its weights | UniVTAC ResNet18 latents of the two marker streams: current frame (clean) + `tactile_future_steps` frames `tactile_future_stride` apart (noised) |

- Attention: video tokens see only video (`first_frame_causal`); action and tactile tokens see the current
  video frame and each other (`tactile_action_attention`).
- Each of action and tactile has its own flow time and loss; the tactile loss covers the valid future frames.
- `freeze_video_expert=true`: the video is the current frame only. Under Fast-WAM's mask no trainable weight
  sees the future frames then, so they would only cost compute (`train_video_future`).
- Tactile encoder frozen, or trainable with EMA targets (`tactile_encoder_trainable`).
- Inference: video prefilled once into a K/V cache, actions and future tactile denoised jointly against it.

The lerobot changes are `patches/` (`git am` on `fe1c545`): 0001/0002 are the Cosmos3 patches of the
UniVTAC repo (0002 adds the per-image-key frame offsets of the dataset loader that the tactile streams use),
0003 is the Fast-WAM tactile expert.

## Setup

```bash
git clone <this repo> ~/Projects/fastwam-univtac
sbatch ~/Projects/fastwam-univtac/scripts/install_cluster.sbatch   # or bash it on a login node
cp <UniVTAC tactile encoder>.pth $WS/fastwam/ckpt/univtac_resnet18_20251128-125750.pth
```

The datasets are the LeRobot datasets of the UniVTAC SmolVLA pipeline
(`$WS/smolvla/data/local/<task>-clean-100_vitac`).

## Train

```bash
sbatch scripts/train_cluster.sbatch lift_bottle
sbatch scripts/train_cluster.sbatch put_bottle_in_shelf
```

Environment variables: `VIDEO=frozen|train`, `TACTILE=none|frozen|trainable`, `TACTILE_STRIDE`,
`TACTILE_STEPS`, `ACTION_HORIZON`, `BATCH_SIZE`, `STEPS`, `SAVE_FREQ`, `GRAD_CKPT`. `RESUME=1` continues the
run from its last checkpoint after the 24 h limit (`--dependency=afterany:<job>`).

## Test

CPU test with a tiny DiT (real Wan VAE and UniVTAC encoder): gradients per expert, tactile init from the
action expert, attention mask, joint sampling.

```bash
cd $WS/fastwam && TACTILE_CKPT=ckpt/univtac_resnet18_20251128-125750.pth HF_HOME=$WS/hf \
    [TRAINABLE=1] [FROZEN_VIDEO=0] .venv/bin/python ~/Projects/fastwam-univtac/tests/test_tactile_expert_tiny.py
```
