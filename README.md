# SDR2HDR_pro

Repository: https://github.com/hajimeno-ipoo/SDR2HDR_pro

公開対象はコード、テスト、依存関係定義、現行の説明文書、実行に必要な最初のTorchScriptモデルです。データセット、学習履歴、比較出力、ONNX書出し、旧設計メモはローカル専用で、リポジトリには含めません。

The repository includes source, tests, dependency definitions, current documentation, and the original TorchScript runtime model. Datasets, training history, comparison outputs, ONNX exports, and historical design notes remain local and are not included.

## 日本語

SDR 動画を HDR10 向けに変換する Python ツールです。GUI と CLI の両方を備えています。

現行版は `AI model 前提` の運用です。変換時には学習済み TorchScript モデル (`.pt`) を指定するか、GUI の `models/` プルダウンから選択する必要があります。

### 現在のモデルと開発目標

使用・改善する学習済みモデルは、最初のモデル `models/enhancement_model_20260310.pt` の1つだけです。同名のONNXファイルと付随データも同じモデルの別形式として保持していますが、GUIとCLIでは`.pt`を使います。

目的は、このモデルの画質をRubyモデル相当へ近づけることです。複数の候補から採用モデルを選ぶことは目的にしません。Ruby相当の画質の達成は未確認です。

v2、保管版、追加候補のモデルと、不要になった比較動画・画像・評価記録は削除済みです。データセットと学習履歴は保持しています。削除前の比較値を現在の品質保証や採用条件として扱いません。

GUIは最初のモデルだけを「使用モデル」として表示し、動画・画像タブで同じモデルを使います。別名のモデルや学習途中のファイルは一覧へ追加しません。最初のモデルがなければ、別モデルへ切り替えずモデル未配置として扱います。CLIの明示的なモデルパス指定と、既存の変換・学習機能は維持しています。

元モデルの学習量不足が画質の原因だったという根拠は未確認です。「Rubyの80〜90%を再現」「元の画を絶対に壊さない」という説明は確認済み仕様ではありません。学習や数値上の誤差低下だけで、最終動画の画質向上を保証しません。

### Overview

- 入力は通常の SDR 動画です。
- 出力は HDR10 メタデータ付きの動画です。
- GUI は queue 実行に対応しています。
- AI モデルは `models/` フォルダ内の `.pt` ファイルを使用します。

### プロ向け出力と品質範囲

- `Apple ProRes 422 HQ` と `Apple ProRes 4444 profile` は10-bit出力です。macOSのVideoToolboxが`p416le`を提供する環境では、`Apple ProRes 4444 XQ`による12-bit出力も利用できます。
- `OpenEXR 16-bit 連番` はBT.2020/PQです。Rubyの線形BT.2020 EXRとは異なります。
- `OpenEXR 16-bit AP1線形連番（表示基準）` は、明るさ調整後の線形RGBをACEScgのAP1原色・ACES白色点へ変換します。中性の値1.0は1.0のままです。撮影時の露光量の復元やACES出力変換の逆変換は行いません。設定キー`openexr_acescg`は維持しています。
- AP1版にはOpenEXRの`exrstdattr`コマンドが必要です。書出し後に原色、白色点、`whiteLuminance`を記録し、コマンドがない場合は変換開始前に停止します。通常は1.0が203 nit、`vivid`では設定したピーク輝度に対応します。`colorInteropID`は`unknown`です。
- 通常EXRはBT.2020/PQのままで、FFmpegは色域属性を記録しません。AP1版は上記の後処理で色域属性を記録します。読み込み先では、通常EXRはBT.2020/PQ、AP1版はAP1/ACES白色点の表示基準の線形値として扱ってください。AP1版を撮影時のシーン露光量として扱った場合の表示一致や、色空間の自動認識は保証しません。
- HDR10のHEVC出力では、処理中に測定したMaxCLL/MaxFALLを使って最終HEVCを再エンコードします。ProResとOpenEXRはHDR10 SEIの対象外です。
- `cinema` はトゥ、ショルダー、ハイライトロールオフ、AIマップ連動の色域拡張を組み合わせたローカルプリセットです。Runway Rubyとの画質同等性は、同一素材・同一表示環境での比較なしには保証しません。

#### 保持している学習データ

LIVEとSol Levanteの元動画、作成済みの学習データ、`data/live_pilot/split.json`の内容分割は保持しています。分割は31内容を学習21・調整5・最終確認5としたものです。最終確認側も過去の評価に使用済みであり、新しい未使用データとしては扱いません。

`scripts/prepare_data.py`は対応するSDR/HDRから学習用の`.npz`を作成します。`content_name`がある場合、学習コードは同じ内容の別解像度・別圧縮率が学習側と検証側に分かれないようにします。新しい実ペアデータには元のBT.2020 HDRも保存します。

3モデルの採用比較、v2固定の時間平滑比較、追加候補専用の学習スクリプトは撤去しました。データ作成、汎用の学習・測定機能、変換計算は維持しています。最初のモデルを改善する具体的な学習方法は、別途コードと画質上の課題を確認して決める段階です。今回の整理では再学習を行っていません。

同じ素材のHDR動画とSDR動画を用意できる場合は、`--sdr-dir`を指定すると人工SDRではなく実写SDRをそのまま教師ペアにできます。両方のディレクトリでファイル名または拡張子を除いた名前を一致させ、フレームレート・フレーム数・解像度も揃えてください。

```bash
uv run python scripts/prepare_data.py \
  --input-dir /path/to/hdr \
  --sdr-dir /path/to/sdr \
  --out-dir /path/to/train_npz \
  --sample-every 24 \
  --peak-nits 1000
```

`--sdr-dir`を省略した場合は、従来どおりHDRから人工SDRを生成します。実写ペアと人工ペアを同じ出力ディレクトリへ作成して混ぜることもできます。

CSVに対応表があるデータセットでは、`--manifest`を指定します。CSVに記載された`video_name`をそのまま使うため、HDRとSDRのファイル名に違いがあっても改名は不要です。`Type=Open-source`の行から、`content_name`・`resolution`・`bitrate`が一致する`HDR10`と`SDR`を一意に対応づけます。

```bash
uv run python scripts/prepare_data.py \
  --input-dir data/LIVE_Paired_Comparison_HDRvsSDR_Database/open-sourced_HDR10 \
  --sdr-dir data/LIVE_Paired_Comparison_HDRvsSDR_Database/open-sourced_SDR \
  --manifest data/LIVE_Paired_Comparison_HDRvsSDR_Database/JOD_separate.csv \
  --out-dir data/live_training/training \
  --sample-every 120 \
  --peak-nits 1000
```

CSVで指定されたファイルがない、対応する形式が片方だけ、または同じキーが重複している場合は、誤ったペアを作らずに停止します。

作成した`.npz`にはCSVの`content_name`を保存します。`scripts/train.py`はこの名前を使い、同じ場面の解像度・ビットレート違いを学習用と検証用へ分けません。CSVを使わない従来形式では、元動画名をグループ名として使います。

元動画が1本だけのデータでは、別の場面を検証用へ回せないため、スクリプトは`frame-fallback (one content group)`と表示して従来のフレーム分割を使います。

### Requirements

- Python
- `ffmpeg` と `ffprobe` が実行可能であること
- PyTorch を含む依存関係

OS ごとの backend は次の通りです。

- Windows: `Auto`, `CUDA`, `CPU / NumPy`
- macOS: `Auto`, `MPS`, `CPU / NumPy`
- その他: `Auto`, `CPU / NumPy`

`Auto` は使える環境で GPU backend を優先し、使えない場合は CPU 側へ寄せます。

### Setup

依存関係をインストールし、学習済みモデルを `models/` に置きます。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

モデル配置例:

```text
models/
  enhancement_model_20260310.pt
```

### Quick Start

#### GUI

```powershell
python -m sdr2hdr.gui
```

GUI の基本動作:

- `Input` と `Output` を指定
- `Preset` は既定で `portrait`
- `AI Model` で `models/` 内の `.pt` を選択
- `AI Strength` は既定で `0.25`
- `Add To Queue` または `Add Files` で queue へ追加
- `Start Queue` で順次変換

#### CLI

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models\enhancement_model_20260310.pt
```

`output_path` を省略した場合は、入力ファイル名の末尾に `_hdr` を付けた名前が自動生成されます。

```powershell
python -m sdr2hdr.cli input.mp4 --model-path models\enhancement_model_20260310.pt
```

### Sample Video

YouTube comparison sample:

[![SDR to HDR comparison sample](https://img.youtube.com/vi/2OFI6urxHxI/maxresdefault.jpg)](https://youtu.be/2OFI6urxHxI)

The sample video compares:

- Left: original SDR footage
- Right: HDR output converted with `sdr2hdr`

### GUI

#### Main Controls

- `Preset`
  - 既定値は `portrait`
- `Encoder`
  - 環境に応じて `libx265`、`NVENC`、`VideoToolbox` を選択
- `Speed/Quality`
  - `Preview`, `Balanced`, `Final`
- `Backend`
  - OS ごとの対応 backend から選択
- `AI Model`
  - `models/enhancement_model_20260310.pt` だけを表示

#### Log メタデータ変換 (Log Metadata Conversion)
- AI処理を介さず、Log素材を直接 HDR10 (BT.2020 / 10-bit HEVC) へ変換するプロフェッショナル向け機能です。
- 階調（見た目）を維持したまま色域のみを拡張するため、編集ソフトでのカラーグレーディングに最適です。

### DaVinci Resolve 連携ガイド

本アプリで生成した HDR10 ファイルを DaVinci Resolve で編集する際の推奨設定です。

1. **プロジェクト設定**:
   - `Color Science`: `DaVinci YRGB Color Managed`
   - `Output Color Space`: `Rec.2020 ST2084 (1000 nits)`
2. **クリップの解釈**:
   - 入力カラースペースに `Rec.2020 ST2084` を指定します。
3. **グレーディング**:
   - ノードの先頭に `Color Space Transform` (Input: Rec.2020/ST2084, Output: DaVinci Wide Gamut/Intermediate) を置くことで、最大限の補正幅を活かした編集が可能になります。
- `Refresh`
  - `models/` を再スキャン
- `AI Strength`
  - 既定値は `0.25`

#### Queue

GUI は複数ジョブの queue 実行に対応しています。

- `Add To Queue`
  - 現在の入力設定を queue に追加
- `Add Files`
  - 複数ファイルをまとめて queue に追加
- `Remove Selected`
  - 選択中の queue 項目を削除
- `Clear Queue`
  - queue を全削除
- `Start Queue`
  - queue を順次処理
- `Stop Current`
  - 実行中ジョブの停止を要求

#### Queue Status

Queue の status 表示は現在次の 7 種類です。

- `QUEUED`
- `STARTING`
- `RUNNING`
- `CANCELLING`
- `OK`
- `FAILED`
- `CANCELLED`

`Stop Current` を押した場合は、まず `CANCELLING` になり、終了時に `CANCELLED` へ確定します。

#### Cancel Behavior

- キャンセル時は partial output を保持する前提です。
- GUI の進捗欄には `partial output saved` と表示されます。

### CLI

現行 CLI の基本仕様:

- `input_path` は必須
- `output_path` は省略可能
- `--model-path` は必須
- `--model-path` は `.pt` モデルを指定
- `--preset` の既定値は `portrait`
- `--backend` は `auto`, `numpy`, `cuda`, `mps`
- `--ai-strength` の既定値は `0.25`

例:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 `
  --preset portrait `
  --backend auto `
  --encoder libx265 `
  --x265-mode balanced `
  --model-path models\enhancement_model_20260310.pt `
  --ai-strength 0.25
```

### Models

- GUI は `models/` フォルダを参照します。
- GUIの対象は `enhancement_model_20260310.pt` のみです。
- モデル未配置時は GUI のプルダウンに有効候補が出ません。
- CLI では `--model-path` に明示指定します。

現在の運用:

- 使用・改善対象は `models/enhancement_model_20260310.pt` の1モデルに固定する
- 削除した候補モデルを選択肢として再追加しない

### Notes

- 現行 README は `利用者向け` の内容に絞っています。
- `peak nits` などの内部パラメータは GUI からは直接設定できません。
- `.onnx` や DirectML は現行の利用手順には含めていません。
- AI モデルなしでの運用は前提にしていません。

### License

This project is licensed under the MIT License.

## English

`sdr2hdr` is a Python tool for converting SDR video to HDR10-style output. It provides both a GUI and a CLI.

The current workflow assumes `AI model usage`. You must provide a trained TorchScript model (`.pt`) for conversion, either from the GUI dropdown or with `--model-path` in the CLI.

### Current model and development goal

The sole retained model for use and improvement is `models/enhancement_model_20260310.pt`. The ONNX files with the same base name are another format of that model, not additional candidates.

The goal is to improve this model toward Ruby-level image quality, not to select between competing models. That quality level has not been verified. The v2, archived v2, and additional candidate model files and obsolete comparison outputs have been removed. Datasets and training history have been retained.

The GUI lists only the retained model, labeled “使用モデル” (model in use). If it is missing, no other model is selected as a substitute. The three-model adoption comparison, fixed-v2 temporal comparison, and candidate-only training scripts have been removed. Dataset preparation, general training and measurement utilities, conversion processing, and explicit CLI model paths are retained.

### Overview

- Input: regular SDR video
- Output: HDR10-tagged video
- The GUI supports queued batch processing
- AI models are loaded from `.pt` files in the `models/` folder

### Requirements

- Python
- `ffmpeg` and `ffprobe` available in `PATH`
- Project dependencies including PyTorch

Backend options by OS:

- Windows: `Auto`, `CUDA`, `CPU / NumPy`
- macOS: `Auto`, `MPS`, `CPU / NumPy`
- Other platforms: `Auto`, `CPU / NumPy`

`Auto` prefers a GPU backend when available and falls back toward CPU processing otherwise.

### Setup

Create a virtual environment, install the package, and place a trained model in `models/`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Example model layout:

```text
models/
  enhancement_model_20260310.pt
```

### Quick Start

#### GUI

```powershell
python -m sdr2hdr.gui
```

Basic GUI workflow:

- Set `Input` and `Output`
- `Preset` defaults to `portrait`
- Select a `.pt` model from `AI Model`
- `AI Strength` defaults to `0.25`
- Add jobs with `Add To Queue` or `Add Files`
- Run them with `Start Queue`

#### CLI

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models\enhancement_model_20260310.pt
```

If `output_path` is omitted, the tool automatically creates a name with `_hdr` appended.

```powershell
python -m sdr2hdr.cli input.mp4 --model-path models\enhancement_model_20260310.pt
```

### Sample Video

YouTube 比較サンプル:

[![SDR→HDR 比較サンプル](https://img.youtube.com/vi/2OFI6urxHxI/maxresdefault.jpg)](https://youtu.be/2OFI6urxHxI)

このサンプル動画では次を比較しています。

- 左: 元の SDR 映像
- 右: `sdr2hdr` で変換した HDR 映像

### GUI

#### Main Controls

- `Preset`
  - Default: `portrait`
- `Encoder`
  - `libx265`, `NVENC`, or `VideoToolbox` depending on platform
- `Speed/Quality`
  - `Preview`, `Balanced`, `Final`
- `Backend`
  - Available backends depend on the current OS
- `AI Model`
  - Lists only `models/enhancement_model_20260310.pt`
- `Refresh`
  - Rescans `models/`
- `AI Strength`
  - Default: `0.25`

#### Queue

The GUI supports multi-job queue execution.

- `Add To Queue`
  - Adds the current form values to the queue
- `Add Files`
  - Adds multiple files to the queue
- `Remove Selected`
  - Removes selected queue items
- `Clear Queue`
  - Clears the queue
- `Start Queue`
  - Starts sequential processing
- `Stop Current`
  - Requests cancellation for the current job

#### Queue Status

The current queue status labels are:

- `QUEUED`
- `STARTING`
- `RUNNING`
- `CANCELLING`
- `OK`
- `FAILED`
- `CANCELLED`

If you press `Stop Current`, the job first moves to `CANCELLING` and then settles on `CANCELLED`.

#### Cancel Behavior

- Partial output is kept on cancellation
- The GUI progress text reports `partial output saved`

### CLI

Current CLI behavior:

- `input_path` is required
- `output_path` is optional
- `--model-path` is required
- `--model-path` must point to a `.pt` model
- `--preset` defaults to `portrait`
- `--backend` supports `auto`, `numpy`, `cuda`, `mps`
- `--ai-strength` defaults to `0.25`

Example:

```powershell
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 `
  --preset portrait `
  --backend auto `
  --encoder libx265 `
  --x265-mode balanced `
  --model-path models\enhancement_model_20260310.pt `
  --ai-strength 0.25
```

### Models

- The GUI scans the `models/` folder
- Only `enhancement_model_20260310.pt` is listed in the GUI
- If no model is present, the GUI dropdown has no usable candidate
- The CLI requires an explicit `--model-path`

Current policy:

- Use and improve only `models/enhancement_model_20260310.pt`
- Do not reintroduce the removed candidate models as alternatives

### Notes

- This README is intentionally user-focused
- Internal parameters such as `peak nits` are not directly exposed in the GUI
- `.onnx` and DirectML are not part of the current usage flow
- Running without an AI model is not the intended workflow

### License

This project is licensed under the MIT License.
