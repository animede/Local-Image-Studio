# Qwen Image 2.1 benchmark: NVIDIA RTX PRO 5000 Blackwell

- Date: 2026-09-25T12:40:49.168265+00:00
- Profile: `fast`
- GPU: NVIDIA RTX PRO 5000 Blackwell (47.26 GiB)
- Driver / CUDA / PyTorch: 580.173.02 / 13.0 / 2.14.0+cu130
- Steps: 40, CFG: 1.0, square output, batch: 1
- Model load: 10.68 s, 10.96 GiB peak

| Case | Output | Ref input | Status | Seconds | Peak VRAM | Torch allocated |
|---|---:|---:|---|---:|---:|---:|
| t2i | 1024² | — | ok | 11.96 | 11.87 GiB | 11.22 GiB |
| t2i | 2048² | — | ok | 76.22 | 14.11 GiB | 12.94 GiB |
| edit1 | 1024² | 1024px | ok | 14.98 | 15.24 GiB | 13.81 GiB |
| edit1 | 2048² | 1024px | ok | 86.72 | 16.89 GiB | 15.54 GiB |
| edit3 | 1024² | 896px | ok | 18.96 | 19.98 GiB | 17.18 GiB |
| edit3 | 2048² | 896px | ok | 98.62 | 21.94 GiB | 18.9 GiB |
| edit10 | 1024² | 512px | ok | 19.77 | 20.87 GiB | 17.73 GiB |
| edit10 | 2048² | 512px | ok | 102.93 | 21.69 GiB | 19.45 GiB |

## Largest successful tested square resolution

- `t2i`: 2048 or higher
- `edit1`: 2048 or higher
- `edit3`: 2048 or higher
- `edit10`: 2048 or higher

> The limit is the largest tested value, not a guaranteed architectural maximum. Desktop/display use and other GPU processes reduce available VRAM.
