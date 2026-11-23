"""FastAPI inference server for the Fast-WAM deploy (run with the Fast-WAM venv, started by deploy_policy.Policy)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request

REPO_ROOT = Path(__file__).resolve().parents[2]


def create_app(authkey: str) -> FastAPI:
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    from policy.fastwam.deploy_policy import _FastWAMLocalPolicy, _from_wire_payload, _to_wire_payload

    app = FastAPI(title="UniVTAC Fast-WAM inference server")
    app.state.model = None

    async def payload_of(request: Request) -> dict:
        payload = await request.json() if request.method == "POST" else {}
        if authkey and request.headers.get("x-fastwam-auth") != authkey and payload.get("authkey") != authkey:
            raise HTTPException(status_code=401, detail="authentication failed")
        return payload

    @app.get("/health")
    async def health(request: Request):
        await payload_of(request)
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/init")
    async def init(request: Request):
        payload = await payload_of(request)
        try:
            if app.state.model is None:
                app.state.model = _FastWAMLocalPolicy(_from_wire_payload(payload["args"]))
            return {"type": "ok"}
        except BaseException as exc:  # noqa: BLE001
            return {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()}

    @app.post("/act")
    async def act(request: Request):
        payload = await payload_of(request)
        if app.state.model is None:
            raise HTTPException(status_code=409, detail="model is not initialized")
        try:
            action = app.state.model.get_action(_from_wire_payload(payload["observation"]), payload.get("instruction"))
            return {"type": "ok", "action": _to_wire_payload(action)}
        except BaseException as exc:  # noqa: BLE001
            return {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()}

    @app.post("/reset")
    async def reset(request: Request):
        await payload_of(request)
        if app.state.model is not None:
            app.state.model.reset()
        return {"type": "ok"}

    @app.post("/shutdown")
    async def shutdown(request: Request):
        await payload_of(request)

        async def stop():
            await asyncio.sleep(0.1)
            app.state.server.should_exit = True

        asyncio.create_task(stop())
        return {"type": "ok"}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast-WAM inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", default="")
    args = parser.parse_args()
    app = create_app(args.authkey)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", access_log=False))
    app.state.server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
