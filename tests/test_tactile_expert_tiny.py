"""CPU unit test of the FastWAM tactile expert with a tiny DiT (real Wan VAE and UniVTAC encoder)."""

import os
import sys

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

torch.manual_seed(0)
TRAINABLE = os.environ.get("TRAINABLE", "0") == "1"
FROZEN_VIDEO = os.environ.get("FROZEN_VIDEO", "1") == "1"
L, H = 2, 256
video = dict(patch_size=[1, 2, 2], in_dim=48, hidden_dim=H, ffn_dim=512, freq_dim=256, text_dim=4096, out_dim=48,
             num_heads=2, attn_head_dim=128, num_layers=L, eps=1e-6, seperated_timestep=True,
             video_attention_mask_mode="first_frame_causal", action_conditioned=False, action_dim=9,
             action_group_causal_mask_mode="group_diagonal")
action = dict(action_dim=9, hidden_dim=128, ffn_dim=256, num_heads=2, attn_head_dim=128, num_layers=L,
              text_dim=4096, freq_dim=256, eps=1e-6)
cams = ["observation.images.head", "observation.images.wrist"]
tac = ["observation.images.tactile_ll_marker", "observation.images.tactile_lr_marker"]
cfg = FastWAMConfig(
    device="cpu", dtype=torch.float32, base_model_id=None, load_text_encoder=False,
    action_dim=9, proprio_dim=16, action_horizon=50 if FROZEN_VIDEO else 48, n_action_steps=50 if FROZEN_VIDEO else 48, image_size=(224, 448),
    num_video_frames=33 if not FROZEN_VIDEO else 33, freeze_video_expert=FROZEN_VIDEO,
    video_dit_config=video, action_dit_config=action, camera_keys=cams, tactile_keys=tac,
    tactile_encoder_path=os.environ["TACTILE_CKPT"], tactile_encoder_trainable=TRAINABLE,
    input_features={**{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in cams},
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,))},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))},
)
print("offsets", cfg.tactile_frame_offsets, "obs delta", cfg.observation_delta_indices,
      {k: cfg.image_delta_indices_for_key(k) for k in cams + tac + ["observation.images.tactile_ll"]})
policy = FastWAMPolicy(cfg)
policy.train()
core = policy.model
print("experts", core.mot.expert_order, "tactile params %.1fM" % (sum(p.numel() for p in core.mot.mixtures["tactile"].parameters()) / 1e6))

# Tactile init from the action expert, as from a released checkpoint without tactile keys.
state = policy.state_dict()
ckpt_keys = {k for k in state if ".tactile." not in k}
policy._init_tactile_from_action(ckpt_keys)
q_a = state["model.mot.layers.0.blocks.action.self_attn.q.weight"]
q_t = state["model.mot.layers.0.blocks.tactile.self_attn.q.weight"]
print("tactile block copied from action:", torch.equal(q_a, q_t))
assert torch.equal(q_a, q_t)

B, T = 2, (1 if FROZEN_VIDEO else 9)
F = len(cfg.tactile_frame_offsets)
batch = {
    "observation.images.head": torch.rand(B, T, 3, 256, 256) if T > 1 else torch.rand(B, 3, 256, 256),
    "observation.images.wrist": torch.rand(B, T, 3, 256, 256) if T > 1 else torch.rand(B, 3, 256, 256),
    "observation.images.tactile_ll": torch.rand(B, 3, 256, 256),  # not a camera: must be ignored
    tac[0]: torch.rand(B, F, 3, 256, 256),
    tac[1]: torch.rand(B, F, 3, 256, 256),
    f"{tac[0]}_is_pad": torch.tensor([[False] * F, [False] * (F - 2) + [True] * 2]),
    f"{tac[1]}_is_pad": torch.tensor([[False] * F, [False] * (F - 2) + [True] * 2]),
    "observation.state": torch.randn(B, 16),
    "action": torch.randn(B, cfg.action_horizon, 9),
    "action_is_pad": torch.zeros(B, cfg.action_horizon, dtype=torch.bool),
    "context": torch.randn(B, 8, 4096),
    "context_mask": torch.ones(B, 8, dtype=torch.bool),
}
loss, metrics = policy.forward(batch)
loss.backward()
print("loss", float(loss), metrics)
assert "loss_tactile" in metrics and torch.isfinite(loss)


def gnorm(prefix):
    g = [p.grad.norm() for n, p in policy.named_parameters() if prefix in n and p.grad is not None]
    return float(torch.stack(g).norm()) if g else 0.0


for name in ["blocks.video", "blocks.action", "blocks.tactile", "mixtures.tactile.action_encoder",
             "mixtures.tactile.stream_embedding", "mixtures.action.action_encoder", "tactile_encoder.",
             "tactile_target_encoder", "proprio_encoder"]:
    print(f"  grad {name}: {gnorm(name):.3e}")
assert gnorm("blocks.tactile") > 0 and gnorm("blocks.action") > 0
assert (gnorm("blocks.video") > 0) == (not FROZEN_VIDEO)
assert (gnorm("tactile_encoder.") > 0) == TRAINABLE and gnorm("tactile_target_encoder") == 0
params = policy.get_optim_params()
n_enc = sum(p.numel() for p in core.tactile_encoder.parameters())
print("optim params %.1fM (tactile encoder %s)" % (sum(p.numel() for p in params) / 1e6,
      "in" if any(p is q for p in params for q in core.tactile_encoder.parameters()) else "out"))
if TRAINABLE:
    opt = torch.optim.SGD(params, lr=1.0)
    opt.step()
    before = [p.clone() for p in core.tactile_target_encoder.parameters()]
    policy.update()
    moved = max(float((a - b).abs().max()) for a, b in zip(core.tactile_target_encoder.parameters(), before))
    print("EMA moved target by", moved)
    assert moved > 0

# Attention mask layout: video | action | tactile.
m = core._build_mot_attention_mask(98, 50, 98, torch.device("cpu"), tactile_seq_len=14)
assert m[98:148, :98].all() and m[98:148, 148:].all() and m[148:, 98:148].all() and not m[:98, 98:].any()
print("mask ok")

# Joint sampling with the video cache (per-sample loop of the policy).
policy.eval()
infer_batch = {k: (v[:, 0] if k in tac or (T > 1 and k in cams) else v) for k, v in batch.items()
               if not k.endswith("_is_pad") and k not in ("action", "action_is_pad")}
infer_batch["context"] = batch["context"][:1]
infer_batch["context_mask"] = batch["context_mask"][:1]
infer_batch = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in infer_batch.items()}
infer_batch["num_inference_steps"] = 3
actions = policy.predict_action_chunk(infer_batch)
print("sampled actions", tuple(actions.shape), "finite", bool(torch.isfinite(actions).all()),
      "tactile", tuple(policy.last_predicted_tactile.shape))
assert actions.shape == (1, cfg.action_horizon, 9) and torch.isfinite(policy.last_predicted_tactile).all()
print("TINY TEST PASSED", "trainable" if TRAINABLE else "frozen", "frozen-video" if FROZEN_VIDEO else "video")
