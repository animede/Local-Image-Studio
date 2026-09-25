from __future__ import annotations

import os
from pathlib import Path


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} は 0/1 または true/false を指定してください。")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    os.getenv("QWEN_MODEL_DIR", PROJECT_ROOT / "models" / "Qwen-Image-2.1")
).expanduser().resolve()
NVFP4_CACHE_DIR = Path(
    os.getenv("QWEN_NVFP4_CACHE_DIR", PROJECT_ROOT / "models" / "Qwen-Image-2.1-nvfp4")
).expanduser().resolve()
VIGGLE_LORA_DIR = Path(
    os.getenv(
        "QWEN_VIGGLE_LORA_DIR",
        PROJECT_ROOT / "models" / "Qwen-Image-2.1-viggle-turbo",
    )
).expanduser().resolve()
OUTPUT_DIR = Path(os.getenv("QWEN_OUTPUT_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
CUDA_DEVICE = os.getenv("QWEN_DEVICE", "cuda:0")
CPU_OFFLOAD = env_flag("QWEN_CPU_OFFLOAD", False)
QUANTIZATION = os.getenv("QWEN_QUANTIZATION", "nvfp4").strip().lower()
if QUANTIZATION not in {"nvfp4", "fp8", "bf16"}:
    raise ValueError("QWEN_QUANTIZATION は nvfp4、fp8、bf16 のいずれかを指定してください。")
START_PROFILE = os.getenv("QWEN_START_PROFILE", "custom").strip().lower()
ACCELERATION = os.getenv("QWEN_ACCELERATION", "none").strip().lower()
if ACCELERATION not in {"none", "viggle-r128"}:
    raise ValueError("QWEN_ACCELERATION は none または viggle-r128 を指定してください。")
REFERENCE_MODE = os.getenv("QWEN_REFERENCE_MODE", "adaptive").strip().lower()
if REFERENCE_MODE not in {"adaptive", "full"}:
    raise ValueError("QWEN_REFERENCE_MODE は adaptive または full を指定してください。")
VAE_TILING = env_flag("QWEN_VAE_TILING", True)
TRIM_CUDA_CACHE = env_flag("QWEN_TRIM_CUDA_CACHE", True)

_default_origins = "http://127.0.0.1:5173,http://localhost:5173"
CORS_ORIGINS = [origin.strip() for origin in os.getenv("QWEN_CORS_ORIGINS", _default_origins).split(",") if origin.strip()]
