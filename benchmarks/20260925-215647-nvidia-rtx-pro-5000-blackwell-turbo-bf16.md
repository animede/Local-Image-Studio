# Qwen Image 2.1 benchmark: NVIDIA RTX PRO 5000 Blackwell

- Date: 2026-09-25T12:56:47.263956+00:00
- Profile: `turbo-bf16`
- GPU: NVIDIA RTX PRO 5000 Blackwell (47.26 GiB)
- Driver / CUDA / PyTorch: 580.173.02 / 13.0 / 2.14.0+cu130
- Steps: 6, CFG: 1.0, square output, batch: 1
- Model load: 6.24 s, 31.24 GiB peak

| Case | Output | Ref input | Status | Seconds | Peak VRAM | Torch allocated |
|---|---:|---:|---|---:|---:|---:|
| t2i | 1024² | — | ok | 4.5 | 32.27 GiB | 31.6 GiB |

## Largest successful tested square resolution

- `t2i`: 1024 or higher

> The limit is the largest tested value, not a guaranteed architectural maximum. Desktop/display use and other GPU processes reduce available VRAM.
