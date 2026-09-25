from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "models" / "Qwen-Image-2.1-viggle-turbo"
DEFAULT_REVISION = "ae38dfbe746f7d027e615e5534c29a16edde9da4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Viggle Turbo r128 for Qwen Image 2.1.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--accept-qwen-research-license",
        action="store_true",
        help="Confirm acceptance of the non-commercial Qwen Research License",
    )
    args = parser.parse_args()
    if not args.accept_qwen_research_license:
        parser.error(
            "read the Qwen Research License and pass --accept-qwen-research-license"
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Viggle/Qwen-Image-2.1-viggle-turbo r128 to {output}")
    path = snapshot_download(
        repo_id="Viggle/Qwen-Image-2.1-viggle-turbo",
        revision=args.revision,
        local_dir=output,
        allow_patterns=[
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
            "scheduler/*",
            "README.md",
            "LICENSE",
            "NOTICE",
        ],
    )
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
