"""Portable, strictly validated learned iterative-reader heads (not RWKV weights)."""
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from safetensors import SafetensorError, safe_open
from safetensors.numpy import save_file

from ..model.iterative_reader import init_controller

FORMAT_NAME = "rwkv7m.iterative_reader"
FORMAT_VERSION = "1"
WEIGHT_SEEDS = (42, 43, 44)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@lru_cache(maxsize=1)
def _parameter_shapes():
    return {name: tuple(value.shape) for name, value in init_controller(0).items()}


def _validate_params(params):
    shapes = _parameter_shapes()
    if not isinstance(params, Mapping) or set(params) != set(shapes):
        raise ValueError("controller tensor names do not match init_controller(0)")
    tensors = {}
    for name, shape in shapes.items():
        value = np.asarray(params[name])
        if value.shape != shape:
            raise ValueError(f"wrong controller tensor shape: {name}: {value.shape} != {shape}")
        if value.dtype != np.dtype(np.float32):
            raise ValueError(f"controller tensor must be FP32: {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite controller tensor: {name}")
        tensors[name] = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    return tensors


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return str(value)


def _validate_metadata(metadata):
    expected = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "dtype": "float32"}
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"unsupported controller {name}: {metadata.get(name)!r}")
    if metadata.get("frontend") not in ("ordinary", "isolated"):
        raise ValueError("controller frontend must be ordinary or isolated")
    if metadata.get("feedback") not in ("true", "false"):
        raise ValueError("controller feedback must be true or false")
    if not _SHA256.fullmatch(metadata.get("weight_sha", "")):
        raise ValueError("controller weight_sha must be a lowercase SHA256 digest")
    for name, minimum in (("weight_seed", 0), ("steps", 0), ("cap", 1)):
        value = metadata.get(name, "")
        if not re.fullmatch(r"0|[1-9][0-9]*", value) or int(value) < minimum:
            raise ValueError(f"invalid controller {name}")
    if int(metadata["weight_seed"]) not in WEIGHT_SEEDS:
        raise ValueError(f"controller weight_seed must be one of {WEIGHT_SEEDS}")


def save_controller(path, params, *, frontend, feedback, weight_seed, weight_sha, steps, cap=6):
    """Save FP32 head tensors and their authenticated-frozen-weight association.

    The source digest is recorded here; the consuming CLI authenticates it against
    the registered frozen checkpoint before any inference.
    """
    if type(feedback) is not bool:
        raise ValueError("feedback must be bool")
    if not isinstance(weight_sha, str):
        raise ValueError("weight_sha must be a SHA256 string")
    metadata = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "dtype": "float32",
        "frontend": frontend,
        "feedback": "true" if feedback else "false",
        "weight_seed": _integer(weight_seed, "weight_seed", 0),
        "weight_sha": weight_sha,
        "steps": _integer(steps, "steps", 0),
        "cap": _integer(cap, "cap", 1),
    }
    _validate_metadata(metadata)
    tensors = _validate_params(params)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    return path


def load_controller(path):
    """Load only the current head schema, returning NumPy FP32 arrays and strings."""
    try:
        with safe_open(str(path), framework="np") as handle:
            metadata = dict(handle.metadata() or {})
            _validate_metadata(metadata)
            tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    except SafetensorError as error:
        raise ValueError(f"invalid controller safetensors artifact: {path}") from error
    return _validate_params(tensors), metadata
