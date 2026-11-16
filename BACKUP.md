# patch-backup: Fast-WAM as patches on upstream LeRobot

This branch keeps the earlier, patch-based version of the Fast-WAM training code, the one the first
runs with the trainable tactile encoder were trained with (lift_bottle, put_bottle_in_shelf,
2026-11-16). `main` has the same changes edited into a trimmed copy of LeRobot instead.

- `patches/`: `git am` on huggingface/lerobot `fe1c545`. 0001/0002 are the Cosmos3 patches (0002 adds
  the per-image-key frame offsets of the dataset loader and the UniVTAC tactile encoder that the
  Fast-WAM tactile expert imports), 0003 is the Fast-WAM tactile expert.
- `scripts/`: install / train sbatch for NHR alex, `sync_wandb.py`. `tests/`: CPU test.

Use: clone this branch and point `REPO` at it (the scripts default to `~/Projects/fastwam-univtac`):

```bash
git clone -b patch-backup git@github.com:Char1esFu/fast-wam.git ~/Projects/fastwam-univtac
sbatch ~/Projects/fastwam-univtac/scripts/install_cluster.sbatch
sbatch ~/Projects/fastwam-univtac/scripts/train_cluster.sbatch lift_bottle
```

The install script clones LeRobot into `$WS/fastwam/lerobot`, applies the patches on branch `univtac`
and builds `$WS/fastwam/.venv`. See README.md for the model and the options.
