"""FastAPI service.

    BHASHAROUTE_MODEL=C_xlmr_finetune_aug uv run uvicorn bhasharoute.serving.api:app --port 8000
    curl -s localhost:8000/route -H 'content-type: application/json' -d '{"text": "lights band kar do"}'
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from bhasharoute.serving.router import Router


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


def create_app(router: Router) -> FastAPI:
    app = FastAPI(title="BhashaRoute", version="0.1.0")

    @app.post("/route")
    def route(req: RouteRequest) -> dict:
        return asdict(router.route(req.text))

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model_version": router.version}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return router.prometheus()

    return app


def _app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Router.from_artifacts(os.environ.get("BHASHAROUTE_MODEL", "C_xlmr_finetune_aug"),
                                            os.environ.get("BHASHAROUTE_DEVICE", "cpu")))


def __getattr__(name: str):  # lazy: importing the module (e.g. in tests) loads no model
    if name == "app":
        return _app_from_env()
    raise AttributeError(name)
