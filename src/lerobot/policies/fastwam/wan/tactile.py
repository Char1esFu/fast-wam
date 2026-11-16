# UniVTAC extension of FastWAM: the tactile expert, a third expert of the MoT.
"""
Tactile expert of the FastWAM MoT (UniVTAC extension).

Tactile tokens are UniVTAC encoder latents, one per tactile stream and frame: the current frame clean (flow time
0) and `tactile_future_steps` future frames, `tactile_future_stride` dataset frames apart, noised and denoised
jointly with the actions. The expert has the architecture of the action expert, and `FastWAMPolicy` starts it
from the action expert's weights, which already read the video expert's keys and values. Its RoPE position is the
frame offset, the time axis of the action tokens (action i is dataset frame t + i); a learned stream embedding
tells the streams apart.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from .model import sinusoidal_embedding_1d
from .modular import ActionDiT


def normalize_tactile_latents(latents: torch.Tensor) -> torch.Tensor:
    """Per-token layer norm without affine: encoder latents to the unit scale of the flow-matching noise."""
    return functional.layer_norm(latents.float(), latents.shape[-1:])


class TactileDiT(ActionDiT):
    """
    Action-expert architecture over tactile latents `[B, F, S, D]` (F frames, S streams, D = `action_dim` =
    the latent width), with one flow time per frame (the current frame stays at 0).
    """

    def __init__(self, num_streams: int, **action_dit_config: Any):
        super().__init__(**action_dit_config)
        self.num_streams = int(num_streams)
        self.stream_embedding = nn.Parameter(torch.zeros(self.num_streams, self.hidden_dim))

    def token_layout(self, frame_offsets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE positions (frame offsets) and stream ids of the F * S tokens, frame-major."""
        positions = torch.tensor(frame_offsets, dtype=torch.long).repeat_interleave(self.num_streams)
        streams = torch.arange(self.num_streams).repeat(len(frame_offsets))
        return positions, streams

    def pre_dit(  # type: ignore[override]
        self,
        tactile_tokens: torch.Tensor,
        token_timesteps: torch.Tensor,
        frame_offsets: list[int],
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Args:
            tactile_tokens: `[B, F, S, D]` latents (noisy for the future frames).
            token_timesteps: `[B, F]` flow time of each frame, in scheduler units.
            frame_offsets: The F frame offsets, in dataset frames from the current frame.
            context, context_mask: Text (and proprio) context of the text cross-attention.
        """
        if tactile_tokens.ndim != 4:
            raise ValueError(f"`tactile_tokens` must be [B, F, S, D], got {tuple(tactile_tokens.shape)}")
        batch_size, num_frames, num_streams, dim = tactile_tokens.shape
        if num_streams != self.num_streams or dim != self.action_dim:
            raise ValueError(
                f"`tactile_tokens` must have {self.num_streams} streams of width {self.action_dim}, "
                f"got {tuple(tactile_tokens.shape)}"
            )
        if len(frame_offsets) != num_frames or tuple(token_timesteps.shape) != (batch_size, num_frames):
            raise ValueError(
                f"{num_frames} tactile frames need {num_frames} frame offsets and [B, F] timesteps, got "
                f"{len(frame_offsets)} and {tuple(token_timesteps.shape)}"
            )
        positions, streams = self.token_layout(frame_offsets)
        if int(positions.max()) >= self.freqs.shape[0]:
            raise ValueError(f"Tactile frame offset {int(positions.max())} exceeds the RoPE cache.")
        seq_len = num_frames * num_streams
        model_dtype = self.action_encoder.weight.dtype

        x = tactile_tokens.reshape(batch_size, seq_len, dim).to(dtype=model_dtype)
        stream_embedding = self.stream_embedding.to(dtype=model_dtype)[streams.to(x.device)]
        tokens = self.action_encoder(x) + stream_embedding
        timesteps = token_timesteps.repeat_interleave(num_streams, dim=1)  # [B, F * S]
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timesteps.reshape(-1)).to(dtype=model_dtype)
        t = self.time_embedding(t_emb).reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))  # per-token modulation

        context_emb = None
        context_attn_mask = None
        if self.use_text_cross_attention:
            if context is None:
                raise ValueError("`context` is required when text cross-attention is enabled.")
            context_emb = self.text_embedding(context.to(dtype=model_dtype))
            if context_mask is None:
                context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
            context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        elif context is not None:
            raise ValueError("Text context was passed to a tactile expert built without text cross-attention.")
        freqs = self.freqs[positions].view(seq_len, 1, -1).to(tokens.device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {"batch_size": batch_size, "seq_len": seq_len, "num_frames": num_frames},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:  # type: ignore[override]
        meta = pre_state["meta"]
        return self.head(tokens).reshape(meta["batch_size"], meta["num_frames"], self.num_streams, self.action_dim)
