# Qwen Image 2.1 高速化・低VRAM化ガイド

この文書は、本プロジェクトで実装した高速化・低VRAM化の要点、起動プロファイルの違い、実測結果、GPU容量別の選択基準をまとめたものです。

## 結論

- Blackwell GPUでは、`turbo`（NVFP4 + Viggle r128）が速度とVRAMの総合最適です。
- 元モデルに近い40-step生成を低VRAMで使う場合は`fast`（NVFP4）を選びます。
- VRAM容量を最優先する場合は`turbo-minimum-vram`、画質確認を含む通常40-stepなら`minimum-vram`です。
- 未量子化BF16を基準に確認する場合は`quality`ですが、1024² T2Iでも約31.6GiB必要です。
- 2-step測定はVRAM限界の確認にしか使えません。速度比較には通常版40 steps、Viggle Turbo 6 stepsを使います。

## 起動プロファイル

| プロファイル | 重み | steps | GPU常駐 | 主な用途 |
|---|---|---:|---|---|
| `quality` | BF16 | 40 | Yes | 未量子化の通常基準 |
| `fast` | NVFP4 W4A4 | 40 | Yes | Blackwellで通常40-stepを高速・低VRAM化 |
| `compatible` | FP8 W8A8 | 40 | Yes | NVFP4非対応環境の互換経路 |
| `minimum-vram` | FP8 W8A8 | 40 | No、CPU offload | 速度よりVRAM最小化 |
| `turbo-bf16` | BF16 + Viggle r128 | 6 | Yes | 非量子化重みで6-step化 |
| `turbo` | NVFP4 + Viggle r128 | 6 | Yes | Blackwellでの推奨・最速構成 |
| `turbo-fp8` | FP8 + Viggle r128 | 6 | Yes | NVFP4非対応環境のTurbo候補 |
| `turbo-minimum-vram` | FP8 + Viggle r128 | 6 | No、CPU offload | Turboを最小VRAMで実行 |

NVFP4とCPU offloadの組み合わせはサポートしていません。

対話メニューを使う場合:

```bash
./start.sh
```

直接指定する場合:

```bash
./start.sh --profile turbo --device cuda:0
./start.sh --profile fast --device cuda:0
./start.sh --profile turbo-minimum-vram --device cuda:0
```

利用可能なプロファイルだけ確認する場合:

```bash
./start.sh --list-profiles
```

## 実測比較

2026-09-25、GPU0のRTX PRO 5000 Blackwell 48GBで測定しました。1024×1024 T2I、CFG 1.0、batch 1、VAE tiling有効です。生成時間はモデルロードを含みません。

| プロファイル | ロード | 生成 | ピークVRAM | `quality`比速度 | Viggle導入効果 |
|---|---:|---:|---:|---:|---:|
| `quality` | 6.03秒 | 20.06秒 | 31.62 GiB | 1.00× | — |
| `fast` | 10.68秒 | 11.96秒 | 11.87 GiB | 1.68× | — |
| `compatible` | 37.67秒 | 18.48秒 | 18.54 GiB | 1.09× | — |
| `minimum-vram` | 27.09秒 | 27.20秒 | 9.80 GiB | 0.74× | — |
| `turbo-bf16` | 6.24秒 | 4.50秒 | 32.27 GiB | 4.46× | 4.46× vs `quality` |
| `turbo` | 12.07秒 | 3.57秒 | 12.59 GiB | 5.62× | 3.35× vs `fast` |
| `turbo-fp8` | 31.39秒 | 4.30秒 | 19.30 GiB | 4.67× | 4.30× vs `compatible` |
| `turbo-minimum-vram` | 28.29秒 | 13.93秒 | 9.81 GiB | 1.44× | 1.95× vs `minimum-vram` |

`quality`比速度は `qualityの生成時間 ÷ 各構成の生成時間` です。値が1より大きければ`quality`より高速、1より小さければ低速です。

測定レポート:

- [fast](../benchmarks/20260925-214049-nvidia-rtx-pro-5000-blackwell-fast.md) / [turbo](../benchmarks/20260925-214218-nvidia-rtx-pro-5000-blackwell-turbo.md)
- [compatible](../benchmarks/20260925-215401-nvidia-rtx-pro-5000-blackwell-compatible.md) / [turbo-fp8](../benchmarks/20260925-215627-nvidia-rtx-pro-5000-blackwell-turbo-fp8.md)
- [quality](../benchmarks/20260925-215437-nvidia-rtx-pro-5000-blackwell-quality.md) / [turbo-bf16](../benchmarks/20260925-215647-nvidia-rtx-pro-5000-blackwell-turbo-bf16.md)
- [minimum-vram](../benchmarks/20260925-215540-nvidia-rtx-pro-5000-blackwell-minimum-vram.md) / [turbo-minimum-vram](../benchmarks/20260925-215738-nvidia-rtx-pro-5000-blackwell-turbo-minimum-vram.md)

### 実用解像度での通常版とTurbo

| 処理 | 出力 | 参照 | `fast` 40-step | `turbo` 6-step | 高速化 |
|---|---:|---:|---:|---:|---:|
| T2I | 1024² | 0 | 11.96秒 / 11.87GiB | 3.57秒 / 12.59GiB | 3.35× |
| T2I | 2048² | 0 | 76.22秒 / 14.11GiB | 15.44秒 / 15.02GiB | 4.94× |
| edit | 1024² | 1 | 14.98秒 / 15.24GiB | 4.01秒 / 15.65GiB | 3.74× |
| edit | 2048² | 1 | 86.72秒 / 16.89GiB | 17.70秒 / 17.88GiB | 4.90× |
| edit | 1024² | 3 | 18.96秒 / 19.98GiB | 5.54秒 / 19.92GiB | 3.42× |
| edit | 2048² | 3 | 98.62秒 / 21.94GiB | 20.59秒 / 22.40GiB | 4.79× |
| edit | 1024² | 10 | 19.77秒 / 20.87GiB | 対象外 | — |
| edit | 2048² | 10 | 102.93秒 / 21.69GiB | 対象外 | — |

TurboはViggleの対象範囲に合わせて参照画像を最大3枚に制限しています。

## 高速化の要点

### 1. BlackwellのNVFP4を使う

`fast`と`turbo`では、DiTとQwen3-VLの主要なLinear層をTorchAOのNVFP4 W4A4へ量子化します。重みだけでなくActivationも4bit化されるため、メモリ帯域とTensor Coreの両方で効果があります。

ただし、次の経路はカーネル要件と形状の都合でBF16のまま維持します。

- DiTの`img_in`
- Qwen3-VLの`model.visual`

無理に全層をNVFP4化すると、非連続Tensorや16の倍数でない幅により実行時エラーになるためです。NVFP4はCompute Capability 10.0以上のBlackwell専用です。

### 2. 事前量子化キャッシュを使う

`scripts/prepare_nvfp4.py`は量子化済みDiTとText Encoderを`models/Qwen-Image-2.1-nvfp4`へ保存します。

- キャッシュ容量: 約10GB
- キャッシュ使用時のロードピーク実測: 約11GiB
- 毎回BF16から量子化する処理と一時ワークスペースを省略

セットアップ時に一度だけ実行します。

```bash
.venv/bin/python scripts/prepare_nvfp4.py
```

### 3. モデルをGPUへ常駐させる

APIサーバーはパイプラインを一度ロードし、生成後もGPUへ常駐させます。通常のリクエストごとにモデルをロードし直さないため、表の「ロード時間」はサーバー起動後の初回ロード時に一度だけ発生します。

GPUジョブは1本のワーカーで直列化し、同時生成によるVRAMピークの重複を防ぎます。

### 4. Viggle r128で6-step化する

Turbo系ではViggle r128 LoRAと専用FlowMatch Euler schedulerを使い、次のsigma scheduleを固定適用します。

```text
[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
```

サーバーは次の条件を自動的に設定します。

- steps: 6
- CFG: 1.0
- 参照画像: 最大3枚

1024²では通常40-step比約3.4倍、2048²では約4.8〜4.9倍高速でした。難しい編集や構図の忠実性では通常40-stepが有利な場合があるため、速度優先時に使用します。

## 低VRAM化の要点

### 1. 重み量子化

1024² T2Iの実測では、BF16の31.62GiBに対しNVFP4は11.87GiBでした。NVFP4 Turboでも12.59GiBです。Blackwellでは、CPU offloadを使わずに重みをGPU常駐させたまま大幅にVRAMを削減できます。

### 2. 参照画像の解像度を自動調整

複数画像editでは、出力解像度ではなくVision Encoderへ入れる参照画像だけを縮小します。

| 参照枚数 | Vision入力解像度 |
|---:|---:|
| 1〜2 | 1024px |
| 3 | 896px |
| 4 | 768px |
| 5 | 704px |
| 6 | 640px |
| 7 | 576px |
| 8〜10 | 512px |

これにより、参照10枚・2048²の通常editでも21.69GiBに収まりました。`QWEN_REFERENCE_MODE=full`を指定すると縮小しませんが、24GBではOOMリスクが高くなります。

### 3. VAE tiling

`QWEN_VAE_TILING=1`でVAEデコードをタイル処理します。特に2048²以上でデコード時の一時VRAMピークを抑えます。わずかな処理オーバーヘッドより、OOM回避効果を優先して全推奨プロファイルで有効にしています。

### 4. 一時CUDAキャッシュの解放

Text/Vision Encoder処理後、denoise開始前に`torch.cuda.empty_cache()`を呼び、不要になった一時キャッシュを返します。複数参照画像では、Vision解析とdenoiseのピークが重ならないことが重要です。

### 5. FP8 CPU offload

`minimum-vram`系では、FP8コンポーネントを必要なときだけGPUへ移します。

- 通常40-step: 9.80GiB、27.20秒
- Viggle 6-step: 9.81GiB、13.93秒

VRAMは最小ですが、PCIe転送が各段階へ加わるためGPU常駐より遅くなります。64GB以上のシステムRAMを推奨します。

## GPU容量別の推奨

| GPU | 推奨 | 判断基準 |
|---:|---|---|
| 16GB Blackwell | `turbo` / `fast` | T2I 2048²は15.02 / 14.11GiB。表示兼用では1024²が安全 |
| 16GB・非Blackwell | `turbo-minimum-vram` | 約9.81GiB。FP8対応状況はGPUごとに確認 |
| 24GB Blackwell | `turbo` | 参照3枚・2048²は22.40GiBで余裕が小さい。通常は参照1枚以下が安全 |
| 32GB | `fast` / `compatible` | BF16 1024²は31.62GiBのため実容量次第で不足 |
| 40GB以上 | `quality` / `turbo-bf16` | BF16をGPU常駐可能 |

GPUの公称容量すべてを推論へ使えるわけではありません。画面表示、デスクトップ、他プロセス、ドライバの使用量として1〜2GiB以上の余裕を見込んでください。

## 設定項目

| 環境変数 | 値 | 内容 |
|---|---|---|
| `QWEN_QUANTIZATION` | `nvfp4` / `fp8` / `bf16` | 重み・Activation形式 |
| `QWEN_ACCELERATION` | `none` / `viggle-r128` | 通常40-stepまたはViggle 6-step |
| `QWEN_CPU_OFFLOAD` | `0` / `1` | CPU offload |
| `QWEN_REFERENCE_MODE` | `adaptive` / `full` | 参照画像解像度調整 |
| `QWEN_VAE_TILING` | `0` / `1` | VAE tiling |
| `QWEN_TRIM_CUDA_CACHE` | `0` / `1` | Encoder後のCUDAキャッシュ解放 |

通常は個別指定ではなく、検証済みプロファイルを使用してください。

## 測定方法

通常40-step:

```bash
.venv/bin/python scripts/benchmark_gpu.py \
  --profile fast \
  --device cuda:0 \
  --resolutions 1024,2048 \
  --cases t2i,edit1,edit3,edit10 \
  --steps 40
```

Viggle 6-step:

```bash
.venv/bin/python scripts/benchmark_gpu.py \
  --profile turbo \
  --device cuda:0 \
  --resolutions 1024,2048 \
  --cases t2i,edit1,edit3
```

レポートは`benchmarks/`へMarkdownとJSONで保存されます。比較時は次を統一してください。

- GPUとデバイス番号
- プロンプト、seed、参照画像
- 出力解像度と参照入力解像度
- 通常版40 steps / Turbo 6 steps
- CFG 1.0、batch 1
- cold/warm条件

`nvidia-smi`のプロセス使用量は実際のGPU占有量、PyTorch allocated/reservedはAllocator内部の値です。本プロジェクトの表では主に`nvidia-smi`のプロセスピークを使用しています。

## 注意事項

- 2-stepの時間を実用速度として比較しないでください。品質が実用条件と異なります。
- ピークVRAMはsteps数にあまり依存しませんが、速度はほぼdenoise回数に依存します。
- Turboは単純に通常版のstepsだけを6へ変更したものではなく、専用LoRAとschedulerが必要です。
- BF16 + Viggleは高速ですが、通常40-stepの`quality`と同じ出力品質を保証するものではありません。
- CPU offloadはVRAMを減らしますが、PCIe帯域とシステムRAMに依存します。
- 別GPUの速度へ単純換算しないでください。Tensor Core、メモリ帯域、電力制限で変わります。

## 実装箇所

- 起動プロファイル: `scripts/start_options.sh`
- モデルロード・量子化・LoRA・低VRAM処理: `backend/app/service.py`
- 環境変数: `backend/app/settings.py`
- NVFP4キャッシュ生成: `scripts/prepare_nvfp4.py`
- Viggle取得: `scripts/download_viggle.py`
- GPUベンチマーク: `scripts/benchmark_gpu.py`
