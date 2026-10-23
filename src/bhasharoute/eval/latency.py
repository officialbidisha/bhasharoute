"""Single-request CPU latency and memory, the numbers that matter for on-device routing."""
from __future__ import annotations

import io
import time

import numpy as np
import torch


def per_request_latency(predict_one, texts: list[str], warmup: int = 10) -> dict:
    for t in texts[:warmup]:
        predict_one(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        predict_one(t)
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "n": len(a)}


def state_dict_mb(module: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell() / 1e6


def dynamic_int8(module: torch.nn.Module) -> torch.nn.Module | None:
    """Post-training dynamic quantisation of every nn.Linear to int8 (weights int8,
    activations quantised on the fly). Returns None if this torch build lacks the API."""
    try:
        from torch.ao.quantization import quantize_dynamic
        torch.backends.quantized.engine = "qnnpack"  # ARM (Apple silicon) kernels
        return quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception as exc:  # pragma: no cover - depends on torch build
        print(f"int8 quantisation unavailable: {exc}")
        return None
