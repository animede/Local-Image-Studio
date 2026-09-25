#!/usr/bin/env python3
"""Benchmark Qwen Image 2.1 speed and VRAM on the selected GPU."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROFILES = {
    "turbo": {
        "QWEN_QUANTIZATION": "nvfp4",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "viggle-r128",
    },
    "turbo-fp8": {
        "QWEN_QUANTIZATION": "fp8",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "viggle-r128",
    },
    "turbo-bf16": {
        "QWEN_QUANTIZATION": "bf16",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "full",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "0",
        "QWEN_ACCELERATION": "viggle-r128",
    },
    "turbo-minimum-vram": {
        "QWEN_QUANTIZATION": "fp8",
        "QWEN_CPU_OFFLOAD": "1",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "viggle-r128",
    },
    "fast": {
        "QWEN_QUANTIZATION": "nvfp4",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "none",
    },
    "compatible": {
        "QWEN_QUANTIZATION": "fp8",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "none",
    },
    "quality": {
        "QWEN_QUANTIZATION": "bf16",
        "QWEN_CPU_OFFLOAD": "0",
        "QWEN_REFERENCE_MODE": "full",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "0",
        "QWEN_ACCELERATION": "none",
    },
    "minimum-vram": {
        "QWEN_QUANTIZATION": "fp8",
        "QWEN_CPU_OFFLOAD": "1",
        "QWEN_REFERENCE_MODE": "adaptive",
        "QWEN_VAE_TILING": "1",
        "QWEN_TRIM_CUDA_CACHE": "1",
        "QWEN_ACCELERATION": "none",
    },
}
VALID_CASES = {"t2i": 0, "edit1": 1, "edit3": 3, "edit10": 10}


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="fast")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolutions", default="1024,1536,2048")
    parser.add_argument("--cases", default="t2i,edit1")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--reference-mode", choices=("adaptive", "full"))
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    args.resolutions = [int(value) for value in csv_values(args.resolutions)]
    args.cases = csv_values(args.cases)
    if not args.resolutions or any(value < 256 or value > 2752 or value % 16 for value in args.resolutions):
        parser.error("resolutions must be 256..2752 and divisible by 16")
    if not args.cases or any(case not in VALID_CASES for case in args.cases):
        parser.error(f"cases must be selected from: {', '.join(VALID_CASES)}")
    if not 2 <= args.steps <= 100:
        parser.error("steps must be 2..100")
    turbo_profile = PROFILES[args.profile]["QWEN_ACCELERATION"] == "viggle-r128"
    if turbo_profile and "edit10" in args.cases:
        parser.error("the turbo profile supports at most 3 reference images")
    return args


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.update(PROFILES[args.profile])
    os.environ["QWEN_START_PROFILE"] = args.profile
    os.environ["QWEN_DEVICE"] = args.device
    if args.reference_mode:
        os.environ["QWEN_REFERENCE_MODE"] = args.reference_mode
    if PROFILES[args.profile]["QWEN_ACCELERATION"] == "viggle-r128":
        args.steps = 6


def process_gpu_memory_mib(pid: int) -> float:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    values = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            values.append(float(fields[1]))
    return max(values, default=0.0)


class MemorySampler:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.values.append(process_gpu_memory_mib(os.getpid()))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.04)

    def __enter__(self) -> "MemorySampler":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop_event.set()
        self.thread.join()

    @property
    def peak_gib(self) -> float | None:
        return round(max(self.values) / 1024, 2) if self.values else None


def make_reference(path: Path | None) -> Image.Image:
    if path:
        with Image.open(path) as image:
            return image.convert("RGB")
    image = Image.new("RGB", (1024, 1024), "#d9e7f5")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 160, 864, 864), fill="#dc624b")
    draw.ellipse((300, 300, 724, 724), fill="#f5d76e")
    draw.text((380, 490), "Qwen benchmark", fill="#172033")
    return image


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def markdown_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"# Qwen Image 2.1 benchmark: {env['gpu_name']}",
        "",
        f"- Date: {report['created_at']}",
        f"- Profile: `{report['profile']}`",
        f"- GPU: {env['gpu_name']} ({env['gpu_memory_gib']} GiB)",
        f"- Driver / CUDA / PyTorch: {env['driver']} / {env['cuda']} / {env['torch']}",
        f"- Steps: {report['steps']}, CFG: 1.0, square output, batch: 1",
        f"- Model load: {report['load']['seconds']} s, {report['load']['peak_vram_gib']} GiB peak",
        "",
        "| Case | Output | Ref input | Status | Seconds | Peak VRAM | Torch allocated |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for result in report["results"]:
        ref = f"{result['reference_resolution']}px" if result["reference_count"] else "—"
        seconds = result.get("seconds", "—")
        peak = result.get("peak_vram_gib")
        allocated = result.get("torch_peak_allocated_gib")
        lines.append(
            f"| {result['case']} | {result['resolution']}² | {ref} | {result['status']} | "
            f"{seconds} | {peak if peak is not None else '—'} GiB | "
            f"{allocated if allocated is not None else '—'} GiB |"
        )
    lines.extend(["", "## Largest successful tested square resolution", ""])
    for case in report["cases"]:
        successful = [
            item["resolution"] for item in report["results"] if item["case"] == case and item["status"] == "ok"
        ]
        tested_max = max(successful) if successful else None
        suffix = " or higher" if tested_max == max(report["resolutions"]) else ""
        lines.append(f"- `{case}`: {tested_max if tested_max else 'none'}{suffix}")
    lines.extend(
        [
            "",
            "> The limit is the largest tested value, not a guaranteed architectural maximum. "
            "Desktop/display use and other GPU processes reduce available VRAM.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_environment(args)

    import torch
    import diffusers
    import torchao
    import transformers

    from backend.app.service import ACCELERATION, VIGGLE_SIGMAS, reference_resolution, service

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[device.index or 0].strip()

    with MemorySampler() as load_sampler:
        load_started = time.perf_counter()
        pipe = service.ensure_model()
        torch.cuda.synchronize(device)
        load_seconds = round(time.perf_counter() - load_started, 2)
    pipe.set_progress_bar_config(disable=True)

    if not args.no_warmup:
        warmup_args: dict[str, Any] = {
            "prompt": "A red sphere on a neutral background",
            "width": 512,
            "height": 512,
            "num_inference_steps": args.steps if ACCELERATION == "viggle-r128" else 2,
            "true_cfg_scale": 1.0,
            "generator": torch.Generator(device=device).manual_seed(args.seed),
        }
        if ACCELERATION == "viggle-r128":
            warmup_args["sigmas"] = VIGGLE_SIGMAS
        warmup_image = pipe(**warmup_args).images[0]
        warmup_image.close()
        gc.collect()
        torch.cuda.empty_cache()

    reference = make_reference(args.reference_image)
    results: list[dict[str, Any]] = []
    for case in args.cases:
        reference_count = VALID_CASES[case]
        for resolution in args.resolutions:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            references = [reference.copy() for _ in range(reference_count)]
            ref_resolution = reference_resolution(reference_count) if references else None
            call_args: dict[str, Any] = {
                "prompt": "A detailed studio photograph of a small red robot on a wooden desk",
                "width": resolution,
                "height": resolution,
                "num_inference_steps": args.steps,
                "true_cfg_scale": 1.0,
                "generator": torch.Generator(device=device).manual_seed(args.seed),
            }
            if references:
                call_args["image"] = references
                call_args["output_resolution"] = ref_resolution
            if ACCELERATION == "viggle-r128":
                call_args["sigmas"] = VIGGLE_SIGMAS
            result: dict[str, Any] = {
                "case": case,
                "resolution": resolution,
                "reference_count": reference_count,
                "reference_resolution": ref_resolution,
            }
            with MemorySampler() as sampler:
                started = time.perf_counter()
                try:
                    image = pipe(**call_args).images[0]
                    torch.cuda.synchronize(device)
                    result.update(status="ok", seconds=round(time.perf_counter() - started, 2))
                    image.close()
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    message = str(exc)
                    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in message.lower():
                        result.update(status="oom", error=message.splitlines()[0])
                    else:
                        result.update(status="error", error=message.splitlines()[0])
            result["peak_vram_gib"] = sampler.peak_gib
            result["torch_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            result["torch_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
            results.append(result)
            print(
                f"{case} {resolution}²: {result['status']}, "
                f"{result.get('seconds', '-')} s, {result['peak_vram_gib']} GiB"
            )
            for image in references:
                image.close()
            if result["status"] == "oom":
                gc.collect()
                torch.cuda.empty_cache()
                break

    created_at = datetime.now(UTC).isoformat()
    report = {
        "created_at": created_at,
        "profile": args.profile,
        "steps": args.steps,
        "seed": args.seed,
        "cases": args.cases,
        "resolutions": args.resolutions,
        "environment": {
            "gpu_name": props.name,
            "gpu_memory_gib": round(props.total_memory / 2**30, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "driver": driver,
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "torchao": torchao.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "load": {"seconds": load_seconds, "peak_vram_gib": load_sampler.peak_gib},
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = args.output_dir / f"{stamp}-{slug(props.name)}-{args.profile}"
    json_path = base.with_suffix(".json")
    markdown_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
