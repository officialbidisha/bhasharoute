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


def is_ffn_output(name: str) -> bool:
    """XLM-R feed-forward down-projection (`encoder.layer.N.output.dense`), not the attention
    output projection (`encoder.layer.N.attention.output.dense`)."""
    return name.endswith(".output.dense") and ".attention." not in name


def int8_layer_names(module: torch.nn.Module) -> list[str]:
    """Encoder nn.Linear layers that tolerate dynamic int8.

    Dynamic quantisation uses one activation scale per tensor. The input to each FFN
    down-projection is the 3072-d GELU output, whose few large outlier channels leave too
    little int8 resolution for everything else. On 500 test requests (frozen encoder, batch 1)
    quantising only those 12 layers cost 6 accuracy points, while the other 60 encoder
    projections together cost <0.5. They stay fp32, as do the small classification heads."""
    return [n for n, m in module.named_modules()
            if isinstance(m, torch.nn.Linear) and n.startswith("encoder.") and not is_ffn_output(n)]


def dynamic_int8(module: torch.nn.Module, inplace: bool = False) -> torch.nn.Module | None:
    """Post-training dynamic quantisation (weights int8, activations quantised on the fly)
    of the layers chosen by `int8_layer_names`. `inplace` avoids holding a second fp32 copy.
    Returns None if this torch build lacks the API."""
    try:
        from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic
        torch.backends.quantized.engine = "qnnpack"  # ARM (Apple silicon) kernels
        spec = {n: default_dynamic_qconfig for n in int8_layer_names(module)}
        return quantize_dynamic(module, spec, dtype=torch.qint8, inplace=inplace)
    except Exception as exc:  # pragma: no cover - depends on torch build
        print(f"int8 quantisation unavailable: {exc}")
        return None
