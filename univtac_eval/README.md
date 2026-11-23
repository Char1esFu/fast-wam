# Evaluating in UniVTAC

The UniVTAC eval harness (`scripts/eval_policy.py` of the UniVTAC repo) runs in the IsaacLab environment; the
policy runs in a venv of this repo. `deploy_policy.Policy` starts `fastwam_server.py` with that venv and talks
to it over HTTP, as the SmolVLA deploy of UniVTAC does.

1. Install this repo into a local venv (here `~/proj/fast-wam/.venv`):

   ```bash
   uv venv ~/proj/fast-wam/.venv --python 3.12
   VIRTUAL_ENV=~/proj/fast-wam/.venv uv pip install -e <this repo> --torch-backend=cu128
   ```

2. Put the files of this directory into `policy/fastwam/` of UniVTAC and set `fastwam_python` in
   `deploy.yml` to that venv's python.

3. The frozen Wan2.2 VAE and UMT5 text encoder load from the HF cache:

   ```bash
   ~/proj/fast-wam/.venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
       d('Wan-AI/Wan2.2-TI2V-5B-Diffusers', allow_patterns=['vae/*', 'text_encoder/*', 'model_index.json']); \
       d('google/umt5-xxl', allow_patterns=['*.json', '*.model'])"
   ```

4. Copy a checkpoint (`checkpoints/last/pretrained_model` of a run, already bfloat16) to
   `{fastwam_ckpt_root}/{task}_{TRAIN_CONFIG}` and run, from the UniVTAC root:

   ```bash
   TRAIN_CONFIG=tac-trainable EP_NUM=100 \
       python scripts/eval_policy.py put_bottle_in_shelf clean fastwam/deploy --headless
   ```

Execution: chunks of 50 actions, the first `fastwam_replan_every` (10) executed, then the next chunk is predicted
(receding horizon). Results go to `eval_result/fastwam/{task}/deploy/{TRAIN_CONFIG}-replan10/`.

Memory: the MoT is 14 GB in bfloat16, 10 GB of it the video expert, too much for a 24 GB GPU next to the simulator
(about 9 GB). The video expert only runs in the per-chunk prefill of the current frame, so with
`fastwam_offload_video` its transformer blocks stay in pinned CPU memory and each is copied to the GPU for its layer
of the prefill: same weights and results, plus the transfer time. The UMT5 text encoder runs on the CPU once per
episode.
