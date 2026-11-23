"""Fast-WAM with touch (world action model of the fast-wam repo) for the UniVTAC eval harness.

As for Cosmos3, the IsaacLab eval process and the policy run in separate Python environments: `Policy` (IsaacLab
env) starts `fastwam_server.py` with the Fast-WAM venv (`fastwam_python`, default ~/proj/fast-wam/.venv) and talks
to it over HTTP; `_FastWAMLocalPolicy` runs in the server.

Checkpoint: `fastwam_ckpt_root/{task}_{TRAIN_CONFIG}` (a lerobot `pretrained_model` dir), e.g.
`TRAIN_CONFIG=tac-trainable` -> ~/proj/fast-wam/ckpt/put_bottle_in_shelf_tac-trainable. A `ckpt_dir` arg overrides it.
The frozen Wan2.2 VAE and UMT5 text encoder come from the HF cache (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`), the
text encoder runs on the CPU once per episode.

Memory: the MoT is 14 GB in bfloat16, 10 GB of it the video expert, which next to the simulator (about 9 GB)
does not fit a 24 GB GPU. The video expert only runs in the per-chunk prefill of the current frame (its K/V are
cached, the action and tactile experts denoise against the cache), so with `fastwam_offload_video` its transformer
blocks stay in pinned CPU memory and each is copied to the GPU for its layer of the prefill. The weights and the
computation are unchanged; only the transfer time is added.

Execution: chunks of `action_horizon` (50) actions, the first `fastwam_replan_every` executed before the next chunk
is predicted (receding horizon, run tag `{TRAIN_CONFIG}-replan{n}`). The policy predicts actions and future
tactile latents jointly; only the actions are used.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_FASTWAM_DIR = Path(__file__).resolve().parent
sys.path.append(str(_FASTWAM_DIR.parent))

from .._base_policy import BasePolicy  # noqa: E402

_DEFAULT_ROOT = Path.home() / "proj" / "fast-wam"


def _deploy_cfg(args: dict) -> dict:
    with open(_FASTWAM_DIR / "deploy.yml", "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in args.items() if k.startswith("fastwam_")})
    return cfg


def _train_config_name(args: dict) -> str:
    return args.get("train_config_name") or os.environ.get("TRAIN_CONFIG") or "tac-trainable"


def _replan_every(args: dict) -> int:
    return int(os.environ.get("FASTWAM_REPLAN_EVERY") or _deploy_cfg(args).get("fastwam_replan_every") or 10)


def get_run_tag(args: dict) -> str:
    """Name of the eval result dir: the checkpoint, plus the execution mode."""
    return f"{_train_config_name(args)}-replan{_replan_every(args)}"


def _instruction_for(task_name: str) -> str:
    """The training task string of the UniVTAC LeRobot datasets (put_bottle_in_shelf -> put the bottle in shelf)."""
    words = task_name.split("_")
    return f"{words[0]} the {' '.join(words[1:])}" if len(words) >= 2 else task_name


def _resolve_ckpt_dir(args: dict) -> Path:
    if args.get("ckpt_dir"):
        return Path(args["ckpt_dir"]).expanduser()
    root = Path(_deploy_cfg(args).get("fastwam_ckpt_root") or _DEFAULT_ROOT / "ckpt").expanduser()
    return root / f"{args['task_name']}_{_train_config_name(args)}"


# ---------------------------------------------------------------------------
# Model runner (Fast-WAM venv, inside fastwam_server.py)
# ---------------------------------------------------------------------------


class _VideoDiTBuiltInDtype:
    """Construct `WanVideoDiT` directly in the policy dtype.

    `_build_core_model` creates the 5B video expert in fp32 and then casts it with `.to(dtype)`, a transient
    host-RAM peak of ~40 GB (fp32 weights plus their random init) that gets the server OOM-killed next to the
    simulator. The init values are overwritten by the checkpoint and the module is cast to `dtype` right after
    either way, so the loaded model is unchanged.
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __enter__(self):
        import torch
        from lerobot.policies.fastwam import modeling_fastwam

        cls = modeling_fastwam.WanVideoDiT
        self._cls, self._init = cls, cls.__init__
        dtype, orig_init = self.dtype, cls.__init__

        def init(module, *args, **kwargs):
            previous = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                orig_init(module, *args, **kwargs)
            finally:
                torch.set_default_dtype(previous)

        cls.__init__ = init
        return self

    def __exit__(self, *exc):
        self._cls.__init__ = self._init
        return False


def _offload_video_blocks(policy, device) -> int:
    """Put everything on `device` except the video transformer blocks (pinned CPU, streamed per prefill layer)
    and a CPU-pinned text encoder. Returns the bytes kept on the CPU."""
    import torch

    core = policy.model
    layers = list(core.mot.layers)
    video_blocks = [layer.blocks["video"] for layer in layers]
    offloaded = {id(t) for b in video_blocks for t in list(b.parameters()) + list(b.buffers())}
    pinned_cpu = set()
    if core.text_encoder is not None and core._text_encoder_device is not None:
        pinned_cpu = {id(t) for t in list(core.text_encoder.parameters()) + list(core.text_encoder.buffers())}

    seen: set[int] = set()
    cpu_bytes = 0
    for module in policy.modules():
        for name, param in module._parameters.items():
            if param is None or id(param) in seen:
                continue
            seen.add(id(param))
            if id(param) in offloaded:
                param.data = param.data.pin_memory()
                param._cpu_data = param.data
                cpu_bytes += param.numel() * param.element_size()
            elif id(param) not in pinned_cpu:
                param.data = param.data.to(device)
        for name, buf in module._buffers.items():
            if buf is None or id(buf) in seen:
                continue
            seen.add(id(buf))
            if id(buf) in offloaded:
                module._buffers[name] = buf.pin_memory()
            elif id(buf) not in pinned_cpu:
                module._buffers[name] = buf.to(device)
    # The frozen VAE is an unregistered attribute (outside `policy.modules()`), so move it explicitly; left on
    # the CPU it encodes every frame there (minutes per call) and hands CPU latents to the CUDA patch embedding.
    core.vae.to(device)
    core.device = torch.device(device)

    def _tensors(block):
        return [p for p in block.parameters()]

    def pre_hook(layer, args, kwargs):
        if kwargs.get("mode") == "video_prefill":
            for p in _tensors(layer.blocks["video"]):
                p.data = p._cpu_data.to(device, non_blocking=True)
            for m in layer.blocks["video"].modules():
                for n, b in m._buffers.items():
                    if b is not None and b.device.type == "cpu":
                        m._buffers[n] = b.to(device, non_blocking=True)

    def post_hook(layer, args, kwargs, output):
        if kwargs.get("mode") == "video_prefill":
            for p in _tensors(layer.blocks["video"]):
                p.data = p._cpu_data

    for layer in layers:
        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        layer.register_forward_hook(post_hook, with_kwargs=True)
    return cpu_bytes


class _FastWAMLocalPolicy:
    # lerobot key -> (observation root, camera, image key)
    _IMAGE_MAP = {
        "observation.images.head": ("observation", "head", "rgb"),
        "observation.images.wrist": ("observation", "wrist", "rgb"),
        "observation.images.tactile_ll": ("tactile", "left_tactile", "rgb"),
        "observation.images.tactile_lr": ("tactile", "right_tactile", "rgb"),
        "observation.images.tactile_ll_marker": ("tactile", "left_tactile", "rgb_marker"),
        "observation.images.tactile_lr_marker": ("tactile", "right_tactile", "rgb_marker"),
    }

    def __init__(self, args: dict):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
        from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors_from_pretrained

        cfg = _deploy_cfg(args)
        self.task_name = args["task_name"]
        self.replan_every = _replan_every(args)
        self.device = torch.device(args.get("device", "cuda"))
        ckpt_dir = _resolve_ckpt_dir(args)
        if not (ckpt_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Fast-WAM checkpoint not found: {ckpt_dir} (set TRAIN_CONFIG or ckpt_dir)")
        print(f"[fastwam] loading {ckpt_dir}", flush=True)

        config = PreTrainedConfig.from_pretrained(str(ckpt_dir))
        offload = bool(cfg.get("fastwam_offload_video", True))
        # Built and loaded on the CPU, then moved (all but the offloaded video blocks): a 14 GB build on the GPU
        # would not fit next to the running simulator.
        config.device = "cpu" if offload else str(self.device)
        config.text_encoder_device = str(cfg.get("fastwam_text_encoder_device") or config.text_encoder_device or "cpu")
        config.tactile_encoder_path = None  # the encoder weights are in the checkpoint
        if cfg.get("fastwam_num_inference_steps"):
            config.num_inference_steps = int(cfg["fastwam_num_inference_steps"])
        self.config = config
        t0 = time.time()
        if config.dtype in (torch.bfloat16, torch.float16) and cfg.get("fastwam_low_mem_build", True):
            with _VideoDiTBuiltInDtype(config.dtype):
                self.model = FastWAMPolicy.from_pretrained(str(ckpt_dir), config=config).eval()
        else:
            self.model = FastWAMPolicy.from_pretrained(str(ckpt_dir), config=config).eval()
        cpu_gb = 0.0
        if offload:
            cpu_gb = _offload_video_blocks(self.model, self.device) / 1e9
            config.device = str(self.device)
        self.pre, self.post = make_fastwam_pre_post_processors_from_pretrained(
            config, ckpt_dir, preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        keys = list(config.camera_keys or [k for k in config.image_features if k not in config.tactile_keys])
        keys += list(config.tactile_keys)
        self._image_map = {k: self._IMAGE_MAP[k] for k in keys}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print(
            f"[fastwam] loaded in {time.time() - t0:.0f}s: {config.dtype}, horizon {config.action_horizon}, "
            f"{config.num_inference_steps} sampling steps, image {config.image_size}, tactile {config.tactile_keys} "
            f"(trainable encoder: {config.tactile_encoder_trainable}); execute {self.replan_every} actions per chunk; "
            f"video blocks offloaded: {cpu_gb:.1f} GB on CPU; GPU allocated "
            f"{torch.cuda.memory_allocated(self.device) / 1e9 if self.device.type == 'cuda' else 0:.1f} GB; "
            f"streams {list(self._image_map)}",
            flush=True,
        )
        self._queue: deque = deque()
        self.instruction = _instruction_for(self.task_name)
        self.chunk_times: list[float] = []

    def encode_obs(self, observation: dict) -> dict:
        import torch

        batch: dict[str, Any] = {}
        for key, (root, cam, img_key) in self._image_map.items():
            img = torch.as_tensor(np.asarray(observation[root][cam][img_key]), device=self.device)
            # HWC uint8 -> CHW [0, 1], as the LeRobot dataset images; the model resizes to its input size.
            batch[key] = img.permute(2, 0, 1).float() / 255.0
        ee = torch.as_tensor(np.asarray(observation["embodiment"]["ee"]), device=self.device).float().view(-1)
        joint = torch.as_tensor(np.asarray(observation["embodiment"]["joint"]), device=self.device).float().view(-1)
        batch["observation.state"] = torch.cat([ee, joint])
        batch["task"] = self.instruction
        return batch

    def get_action(self, observation: dict, instruction: str | None = None) -> np.ndarray:
        import torch

        if self._queue:
            return self._queue.popleft()
        batch = self.pre(self.encode_obs(observation))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            raw_chunk = self.model.predict_action_chunk(batch)
        chunk = self.post(raw_chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.chunk_times.append(time.time() - t0)
        if len(self.chunk_times) in (1, 2, 10) or len(self.chunk_times) % 100 == 0:
            peak = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else 0
            print(f"[fastwam] chunk {len(self.chunk_times)}: {self.chunk_times[-1]:.2f}s, peak {peak:.1f} GB", flush=True)
        chunk = torch.as_tensor(chunk, dtype=torch.float32).reshape(-1, chunk.shape[-1])
        self._queue.extend(a.cpu().numpy() for a in chunk[: self.replan_every])
        return self._queue.popleft()

    def reset(self):
        self._queue.clear()
        self.model.reset()


# ---------------------------------------------------------------------------
# HTTP transport (both environments)
# ---------------------------------------------------------------------------


def _to_wire_payload(value: Any):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": str(array.dtype), "shape": list(array.shape),
                "data": base64.b64encode(array.tobytes()).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): _to_wire_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire_payload(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return _to_wire_payload(np.asarray(value))


def _from_wire_payload(value: Any):
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            data = base64.b64decode(value["data"].encode("ascii"))
            return np.frombuffer(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
        return {k: _from_wire_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_wire_payload(v) for v in value]
    return value


class _FastWAMClient:
    """Starts fastwam_server.py with the Fast-WAM venv on a free local port (or uses `fastwam_port`)."""

    def __init__(self, args: dict):
        cfg = _deploy_cfg(args)
        self.host = str(cfg.get("fastwam_host", "127.0.0.1"))
        self.port = int(cfg.get("fastwam_port") or 0)
        self.python = Path(cfg.get("fastwam_python") or _DEFAULT_ROOT / ".venv" / "bin" / "python").expanduser()
        self.timeout = float(cfg.get("fastwam_request_timeout", 600))
        self.authkey = str(cfg.get("fastwam_authkey", ""))
        self._process: subprocess.Popen | None = None
        self._external = self.port != 0
        if not self._external:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = int(sock.getsockname()[1])
            self.authkey = secrets.token_hex(16)
            if not self.python.exists():
                raise FileNotFoundError(f"Fast-WAM python not found: {self.python}")
            self._process = subprocess.Popen(
                [str(self.python), str(_FASTWAM_DIR / "fastwam_server.py"), "--host", self.host,
                 "--port", str(self.port), "--authkey", self.authkey],
                cwd=str(_FASTWAM_DIR.parents[1]),
                # A PYTHONPATH to another lerobot (SmolVLA's src) would shadow the Fast-WAM one.
                env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            )
        self.base_url = f"http://{self.host}:{self.port}"
        deadline = time.monotonic() + float(cfg.get("fastwam_startup_timeout", 300))
        while True:
            try:
                if self._request("GET", "/health").get("status") == "ok":
                    break
            except Exception as exc:  # noqa: BLE001
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"Fast-WAM server exited with code {self._process.returncode}") from exc
                if time.monotonic() > deadline:
                    raise TimeoutError("timed out waiting for the Fast-WAM server") from exc
            time.sleep(0.5)
        self._post("/init", {"args": _to_wire_payload(args)})

    def _request(self, method: str, path: str, payload: Any = None):
        data = None if payload is None else json.dumps({**payload, "authkey": self.authkey}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"X-FastWAM-Auth": self.authkey, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Fast-WAM server HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def _post(self, path: str, payload: Any):
        response = self._request("POST", path, payload)
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(f"Fast-WAM server error:\n{response.get('error')}\n{response.get('traceback')}")
        return response

    def get_action(self, observation: dict, instruction: str | None) -> np.ndarray:
        response = self._post("/act", {"instruction": instruction, "observation": _to_wire_payload(observation)})
        return np.asarray(_from_wire_payload(response["action"]), dtype=np.float32)

    def reset(self) -> None:
        self._post("/reset", {})

    def close(self) -> None:
        try:
            self._post("/reset" if self._external else "/shutdown", {})
        except Exception:  # noqa: BLE001
            pass
        if self._process is not None:
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()


class Policy(BasePolicy):
    """Deploy-time Fast-WAM wrapper for UniVTAC (IsaacLab side)."""

    def __init__(self, args: dict):
        self.task_name = args["task_name"]
        self.model = _FastWAMClient(args)

    def eval(self, task, observation):
        import torch

        action = self.model.get_action(observation, None).reshape(-1)
        # Trained on embodiment/joint: 7 arm joints + 2 mirrored finger joints; take_action("qpos") takes
        # 7 arm joints + 1 gripper value.
        action_t = torch.from_numpy(action[:8].copy()).to(task.device).float()
        return task.take_action(action_t, action_type="qpos")

    def reset(self):
        self.model.reset()

    def close(self):
        self.model.close()
