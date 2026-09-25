#!/usr/bin/env python3
"""Build reusable TorchAO NVFP4 component checkpoints for low-peak startup."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "models" / "Qwen-Image-2.1",
        help="Source Diffusers model directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "Qwen-Image-2.1-nvfp4",
        help="Destination for quantized transformer/text encoder",
    )
    parser.add_argument("--device", default="cuda:0", help="Blackwell GPU used for conversion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not (source / "model_index.json").is_file():
        raise SystemExit(f"Source model not found: {source}")
    if (output / ".complete").is_file():
        print(f"NVFP4 cache already exists: {output}")
        return

    import torch
    from diffusers import QwenImage21Transformer2DModel
    from diffusers import TorchAoConfig as DiffusersTorchAoConfig
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() < (10, 0):
        raise SystemExit("NVFP4 conversion requires a Blackwell GPU")

    output.mkdir(parents=True, exist_ok=True)
    print("Quantizing transformer...")
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        source / "transformer",
        dtype=torch.bfloat16,
        local_files_only=True,
        quantization_config=DiffusersTorchAoConfig(
            NVFP4DynamicActivationNVFP4WeightConfig(),
            modules_to_not_convert=["img_in"],
        ),
        device_map=args.device,
    )
    transformer.save_pretrained(output / "transformer", safe_serialization=True)
    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    print("Quantizing text encoder...")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        source / "text_encoder",
        dtype=torch.bfloat16,
        local_files_only=True,
        quantization_config=TransformersTorchAoConfig(
            NVFP4DynamicActivationNVFP4WeightConfig(),
            modules_to_not_convert=["model.visual"],
        ),
        device_map=args.device,
    )
    text_encoder.save_pretrained(output / "text_encoder", safe_serialization=True)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    (output / ".complete").write_text("torchao-nvfp4\n", encoding="utf-8")
    print(f"NVFP4 cache created: {output}")


if __name__ == "__main__":
    main()
