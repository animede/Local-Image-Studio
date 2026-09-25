# Qwen Image 2.1 benchmark: NVIDIA RTX PRO 5000 Blackwell

- Date: 2026-09-25T12:56:27.678739+00:00
- Profile: `turbo-fp8`
- GPU: NVIDIA RTX PRO 5000 Blackwell (47.26 GiB)
- Driver / CUDA / PyTorch: 580.173.02 / 13.0 / 2.14.0+cu130
- Steps: 6, CFG: 1.0, square output, batch: 1
- Model load: 31.39 s, 17.63 GiB peak

| Case | Output | Ref input | Status | Seconds | Peak VRAM | Torch allocated |
|---|---:|---:|---|---:|---:|---:|
| t2i | 1024² | — | ok | 4.3 | 19.3 GiB | 18.28 GiB |

## Largest successful tested square resolution

- `t2i`: 1024 or higher

> The limit is the largest tested value, not a guaranteed architectural maximum. Desktop/display use and other GPU processes reduce available VRAM.
