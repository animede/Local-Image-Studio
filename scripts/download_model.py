from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "models" / "Qwen-Image-2.1"
DEFAULT_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official Diffusers Qwen-Image-2.1 model.")
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
    print(f"Downloading Qwen/Qwen-Image-2.1 to {output}")
    path = snapshot_download(
        repo_id="Qwen/Qwen-Image-2.1",
        revision=args.revision,
        local_dir=output,
    )
    (output / ".download-complete").write_text(
        f"repo=Qwen/Qwen-Image-2.1\nrevision={args.revision}\n", encoding="utf-8"
    )
    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
