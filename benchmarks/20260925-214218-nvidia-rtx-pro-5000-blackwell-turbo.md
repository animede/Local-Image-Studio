# Qwen Image 2.1 benchmark: NVIDIA RTX PRO 5000 Blackwell

- Date: 2026-09-25T12:42:18.111945+00:00
- Profile: `turbo`
- GPU: NVIDIA RTX PRO 5000 Blackwell (47.26 GiB)
- Driver / CUDA / PyTorch: 580.173.02 / 13.0 / 2.14.0+cu130
- Steps: 6, CFG: 1.0, square output, batch: 1
- Model load: 12.07 s, 11.62 GiB peak

| Case | Output | Ref input | Status | Seconds | Peak VRAM | Torch allocated |
|---|---:|---:|---|---:|---:|---:|
| t2i | 1024² | — | ok | 3.57 | 12.59 GiB | 11.94 GiB |
| t2i | 2048² | — | ok | 15.44 | 15.02 GiB | 13.91 GiB |
| edit1 | 1024² | 1024px | ok | 4.01 | 15.65 GiB | 14.62 GiB |
| edit1 | 2048² | 1024px | ok | 17.7 | 17.88 GiB | 16.6 GiB |
| edit3 | 1024² | 896px | ok | 5.54 | 19.92 GiB | 18.1 GiB |
| edit3 | 2048² | 896px | ok | 20.59 | 22.4 GiB | 20.07 GiB |

## Largest successful tested square resolution

- `t2i`: 2048 or higher
- `edit1`: 2048 or higher
- `edit3`: 2048 or higher

> The limit is the largest tested value, not a guaranteed architectural maximum. Desktop/display use and other GPU processes reduce available VRAM.
