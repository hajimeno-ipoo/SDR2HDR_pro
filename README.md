# SDR2HDR Pro

SDR2HDR Proは、SDR動画・静止画をHDRへ変換し、変換前後を比較してから指定形式で保存するローカルアプリです。

GUIは「変換」「書き出し」「比較」の3つの欄で構成されています。変換と書き出しは別の操作で、変換完了だけでは指定した保存先にファイルは作られません。動画AI変換はCLIからも実行できます。

本体はPythonとTkinterで動作します。変換にはFFmpeg、AI処理には同梱のTorchScriptモデルを使用し、変換処理を外部サービスへ送信しません。

元作者・元プロジェクト：[jpneagle / sdr2hdr](https://github.com/jpneagle/sdr2hdr)

## 4つの変換モード

| GUIのタブ | 対象 | 処理 |
| --- | --- | --- |
| 動画 AI | SDR動画 | AIモデルとプリセットを使ってHDRへ変換 |
| 動画 Log | 動画 | AIを使わず、入力をBT.709として色と明るさの表現を変換 |
| 画像 AI | SDR静止画 | AIモデルとプリセットを使ってHDR静止画へ変換 |
| 画像 Log | 静止画 | AIを使わず、入力をBT.709としてBT.2020/PQへ変換 |

**「Log」は現在の画面上の名称です。カメラ固有のLog曲線を選択・復号する機能ではありません。** この経路はBT.709入力と100 nit基準を前提に画素を変換します。色情報のラベルだけを書き換える処理とも異なります。

HLGと、ACES 1.3／2.0の版を指定するACEScg出力は「動画 AI」で選択できます。「動画 Log」では選択できません。

## 主な機能

- 1件または複数の素材を選択し、ファイルごとに設定を持つ変換一覧へ登録
- 動画・静止画のAI変換と、AIを使わない変換
- HEVC/PQ、HEVC/HLG、ProRes、OpenEXR、5種類のHDR静止画形式への保存
- プリセット、AI強度、彩度、折りたたみ式の詳細HDR設定
- SDR/HDRの上下比較、動画の同期再生・シーク・コマ送り・ミュート
- 固定倍率の選択、＋／−による微調整、拡大中の表示位置移動
- 変換後の明示的な書き出し、保存成功分の一覧・一時ファイルの自動整理
- 進捗表示、処理の停止、別ウインドウでの処理ログ確認

利用できる形式や処理方式は、OS、GPU、FFmpegの構成によって異なります。

## 画面構成と使い方

### 01 / CONVERT：素材と変換設定

1. 動画・画像、AI・Logのタブを選びます。
2. 入力の「参照」から、1件または複数の素材を選びます。
3. 出力先と出力形式を指定します。AIモードではプリセットなども設定します。
4. 「変換待ちに追加」で一覧へ登録します。
5. 「変換を開始」で、待機中の素材を順番に変換します。一覧が空の場合は、現在の入力を登録して開始します。

一覧に追加した時点の設定を各ファイルへ保持します。その後に画面の設定を変更しても、登録済みの項目は変更しません。

#### 入力と出力名

入力の参照では、動画はMP4・MOV・MKV・M2TS・MTS、静止画はJPEG・PNG・TIFF・WebP・HEIC・AVIFを絞り込み表示します。「すべてのファイル」でも選択できますが、読み込み可否はデコーダーの対応状況に依存します。

- 1件ずつ選ぶ場合は、出力ファイル名を指定します。別の素材へ切り替えると、保存先フォルダーを保ち、新しい素材名に `_hdr` と出力形式の拡張子を付けます。
- 同じ素材を選び直した場合や出力形式だけを変更した場合は、手動で付けた名前を保持します。
- 複数件を一度に選ぶ場合は、共通の保存先フォルダーを指定します。各素材名から出力名を作り、既存ファイルや一覧の保存先と重なると番号を付けます。
- 個別登録で同じ保存先を指定した場合は、名前の変更を求めて追加を止めます。書き出し直前にも、対象項目の保存先が重複していないか確認します。

#### AI設定

プリセットは `natural`（初期設定）、`balanced`、`high`、`portrait`、`anime`、`cinema`、`poc` の7種類です。明るさの拡張、細部、保護処理などの変換設定をまとめて切り替えます。`poc` は処理速度を優先した検証用設定です。

基本欄では、処理方式、使用モデル、AI強度、彩度を指定できます。「▶ 詳細HDR設定」を開くと次の項目を操作できます。

| 項目 | 内容 |
| --- | --- |
| 目標ピーク輝度 | HDR変換の目標となる最大輝度。単位はnit |
| HDR Guidance | 自動 / ON / OFF。対応モデルが返す目標輝度と復元情報の利用を切り替え |
| 輝度ガイダンス | モデルの目標輝度を反映する強さ |
| 復元強度 | 白飛び領域に対するモデルの復元情報を反映する強さの上限 |

詳細欄は初期状態で閉じています。閉じても設定は維持します。プリセットを切り替えると、目標ピーク輝度・AI強度・彩度もそのプリセットの値へ更新します。AI設定は動画AIと画像AIで共有します。

HDR Guidanceの「自動」は、モデルが追加の目標輝度・復元情報を返す場合にそれを使い、「OFF」は使いません。動画AIで「ON」を選び、モデルがこの追加情報に対応していなければエラーになります。失われた細部の復元を保証する機能ではありません。

HLGと版付きACEScg出力では、目標ピーク輝度を0超〜1000 nit以下に設定する必要があります。

### 02 / EXPORT：変換結果と書き出し

「変換を開始」はアプリ専用の一時領域に結果を作ります。一覧の「未書き出し」は、変換は終わっているものの、指定先にはまだ保存していない状態です。

「変換済みを書き出す（件数）」を押すと、変換が完了している未書き出し分を各指定先へ保存します。この段階はコピーだけで、映像・画像・音声を再圧縮しません。保存先に既存ファイルがある場合は、上書きするか確認します。

書き出し処理が終わると、保存に成功した項目だけをEXPORT・COMPAREの両一覧から取り除き、対応する比較再生・表示を解除して一時ファイルを削除します。元素材と保存済みファイルは削除しません。

- 「出力を開く」：最後に書き出したファイルを開きます。
- 「保存フォルダ」：最後に書き出したファイルの保存先フォルダーを開きます。
- 「処理ログを開く」：処理内容やエラーを確認する小窓を開きます。

一覧が空になっても、保存先を開く2つのボタンは使用できます。書き出し済みの結果を見返す場合は、保存ファイルを開いてください。

### 03 / COMPARE：変換前後の比較

上段に入力SDR、下段に変換後HDRを表示します。複数の変換結果がある場合は、上部の一覧で切り替えます。各画像の下に、取得できた色空間・伝達特性・ビット深度を表示します。情報を取得できない項目は `Unknown`、アプリの出力設定に基づく情報は「変換設定」と表示します。

動画には次の操作があります。

- 先頭へ移動、前後のコマ送り、再生・停止
- シークバーを動かして再生位置を変更。ドラッグ中も移動先の表示を更新
- ミュートの切り替え。比較音声はSDR側だけから再生し、二重再生を避けます

静止画では動画用の操作を隠します。OpenEXR連番のZIPは比較一覧に登録しません。EXRからHDR動画として保存する設定を選んだ場合は、変換後のMOVを比較できます。

#### 拡大縮小と表示位置

「共通倍率」のプルダウンで25%〜400%の固定倍率を選べます。隣の＋／−ボタンでは1パーセントポイントずつ微調整できます。

100%は画像全体が表示枠に収まる大きさで、元画像の1画素と画面の1画素を一致させる表示ではありません。倍率と表示位置はSDR/HDRの両側で共通です。100%を超える場合は、画像をドラッグして表示位置を移動できます。

「全体表示」と素材の切り替えで100%へ戻ります。倍率変更時は表示位置を中央へ戻します。これらの操作は比較表示だけに適用し、保存するファイルは変更しません。

## 書き出し仕様

### 動画

| 出力 | CLIの `--encoder` | 保存形式・条件 |
| --- | --- | --- |
| HEVC / PQ（CPU） | `libx265` | 10bit、BT.2020/PQ、MP4・MOV・MKV |
| HEVC / PQ（Mac） | `hevc_videotoolbox` | 10bit、BT.2020/PQ、対応Mac |
| HEVC / PQ（NVIDIA） | `hevc_nvenc` | 10bit、BT.2020/PQ、対応NVIDIA環境 |
| HEVC / HLG | `hevc_hlg` | 10bit、BT.2020/HLG、基準ピーク1000 nit |
| ProRes 422 HQ | `prores_422hq` | MOV、10bit、BT.2020/PQ |
| ProRes 4444 | `prores_4444` | MOV、このアプリの入力精度は10bit、BT.2020/PQ |
| ProRes 4444 XQ | `prores_4444_xq` | MOV、12bit出力経路。対応Mac・VideoToolbox・FFmpegが必要 |
| OpenEXR（PQ） | `openexr` | BT.2020/PQのhalf float連番 |
| OpenEXR（表示基準AP1線形） | `openexr_acescg` | 表示基準のAP1線形値。版付きACEScgとは別の出力 |
| OpenEXR ACEScg（ACES 1.3） | `openexr_acescg_1_3` | ACES 1.3用の公式出力変換を逆方向に適用 |
| OpenEXR ACEScg（ACES 2.0） | `openexr_acescg_2_0` | ACES 2.0用の公式出力変換を逆方向に適用 |

GUIはOSに応じてMac用・NVIDIA用の選択肢を表示します。選択肢の表示だけで、GPUやエンコーダーが利用可能と保証するものではありません。ProRes 4444 XQは必要なエンコーダーと入力形式を検出できなければエラーになります。

#### OpenEXRの保存方法

GUIの「保存方法」、またはCLIの `--exr-delivery` で選びます。

- `zip`（初期設定）：half floatのEXR連番を1つのZIPへ保存します。入力に音声があれば `audio.mov` または `audio.mka` も同梱します。
- `video`：EXR連番をBT.2020/PQへ変換し、音声とともにProRes 4444 / MOVへ保存します。出力先の拡張子は `.mov` にします。

**ACEScgやAP1線形のデータを保持したい場合はZIPを選んでください。** 「HDR動画」はACEScgのままの動画ではありません。版付きACEScgからの動画化には、選択したACES版の公式出力変換を使います。

表示基準AP1線形と版付きACEScgのEXRには、`exrstdattr` コマンドが必要です。版付きACEScgのEXRには、ACESの版・変換情報も記録します。通常の `openexr` は線形RGBではなくPQです。レビュー用の線形EXR抽出とは区別してください。

### 静止画

画像AI・画像Logの両方で選択できます。

| 出力 | 拡張子 | 保存内容 |
| --- | --- | --- |
| TIFF | `.tif` / `.tiff` | RGB各16bit、BT.2020/PQ。BT.2100 PQのICCプロファイルを埋め込み |
| JPEG XL | `.jxl` | RGB各16bit、BT.2020/PQ。変換後の画素をロスレス圧縮 |
| AVIF | `.avif` | 10bit・4:2:0、BT.2020/PQ。非可逆圧縮 |
| Ultra HDR JPEG | `.jpg` / `.jpeg` | SDR画像と、HDR復元用の明るさの補助情報（ゲインマップ）。10bit PQ入力から生成、非可逆圧縮 |
| PNG | `.png` | RGB各16bit、ロスレス。cICP情報でBT.2020/PQ・RGB・フルレンジを記録 |

Ultra HDR JPEGの保存には `ultrahdr_app` が必要です。非対応の閲覧アプリではSDR画像を表示します。TIFFのICCには、macOSではシステムのPQプロファイル、それ以外では同梱プロファイルを使用します。

ここでの「ロスレス」は、HDR変換後の画素を保存するときの圧縮方式を指します。元のSDR画像との画素の完全一致を意味しません。

### 色情報と表示の注意

編集ソフトへ読み込む際の設定は、[HDR動画を編集ソフトで扱うためのカラー設定](HDR_EDITING_GUIDE.md)を参照してください。ソフト別の一覧表、PQとHLGの設定の違い、Macの表示設定をまとめています（参照会話の整理資料）。

- HDRの見え方を確認するには、HDR対応画面、OS設定、対応する閲覧アプリが必要です。
- macOSの動画比較はlibmpvとネイティブ表示面を使用します。表示できる明るさの範囲に応じて表示用の輝度を調整します。
- macOSの静止画比較は `NSImageView` で元ファイルを直接読み、ICCプロファイルとゲインマップをmacOS標準の画像表示へ渡します。動画用の表示処理とは別です。
- Windowsの比較表示はlibmpvのD3D11経路です。Windows実機でのHDR表示は未確認です。Linuxのアプリ内比較プレイヤーは未対応です。
- 形式や閲覧アプリによって表示用の明るさ調整が異なるため、見た目の完全一致は保証しません。比較表示の設定は保存済みファイルを変更しません。
- HEVC/PQでは、`verify_hdr_metadata=True` のとき色情報を書き直した後、ffprobeでBT.2020・PQ・BT.2020ncを読み戻して確認します。この検査はMaxCLL・MaxFALL・マスタリング情報すべての検証ではありません。
- ProResは映像内部とMOVにBT.2020/PQの色情報を記録します。
- 版付きACEScg出力は、HDR変換後の映像に対する公式出力変換の逆変換です。カメラが撮影時に受けた光や、白飛びで失われた情報の復元を保証しません。編集ソフトでは出力に合った入力色空間を設定してください。
- 版付きACEScgとHLGの出力変換は1000 nit基準です。公式変換の切り詰めや色域処理を保持しており、往復変換で元の数値へ完全一致することを保証しません。

出力用の色変換は [output_color.py](src/sdr2hdr/output_color.py)、保存処理は [io.py](src/sdr2hdr/io.py) を参照してください。

## 一時ファイルと停止時の扱い

GUIの未書き出し結果はアプリ専用の一時ファイルです。作業の再開用に、次回起動まで保存する仕組みではありません。

- **変換を停止**：今回の未完成出力を破棄します。ほかの変換完了済みの結果は保持します。
- **書き出しを停止・失敗**：保存に成功した分は残します。まだ保存していない変換結果は保持し、再度書き出せます。失敗・中断したコピーで既存の保存ファイルを置き換えません。
- **書き出し成功**：成功分だけ両一覧と一時ファイルを整理します。比較解除や一時ファイル削除に失敗した場合は結果を残し、ログで知らせます。
- **通常のアプリ終了**：処理と再生を停止してから、一時領域を削除します。未書き出し結果も失われるため、必要な結果は終了前に書き出してください。
- 強制終了や電源断の後に、一時ファイルが自動で削除されることは保証しません。

## 必要な環境とインストール

### 必要なもの

- Python 3.11以上。GUIにはTkinterが必要です。
- FFmpegとffprobe。変換用の `zscale` フィルターと、使う形式のエンコーダーを含む構成が必要です。
- AIモードにはPyTorchとTorchScriptモデル。
- 動画比較にはlibmpv。Pythonの `python-mpv` パッケージとは別に必要です。
- macOSではXcode Command Line Tools。インストール時にネイティブ表示面をビルドします。静止画HDR表示にはmacOSの対応APIが必要です。
- AP1系EXRには `exrstdattr`、Ultra HDR JPEGには `ultrahdr_app`。

macOS・Windows向けの比較表示経路がありますが、OSや依存ライブラリの全バージョンでの動作を保証するものではありません。Linuxではアプリ内比較を利用できません。

### macOSの外部ツール

Homebrewを利用する場合の導入例です。

```bash
xcode-select --install
brew install ffmpeg-full mpv
export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"
```

必要な形式に応じて追加します。

```bash
# AP1線形・ACEScgのEXR
brew install openexr

# Ultra HDR JPEG
brew install libultrahdr
```

`ffmpeg-full` は通常の `ffmpeg` とは別の配置になるため、起動するターミナルでPATHを設定してください。配布内容はHomebrewの [ffmpeg-full](https://formulae.brew.sh/formula/ffmpeg-full)、[mpv](https://formulae.brew.sh/formula/mpv)、[OpenEXR](https://formulae.brew.sh/formula/openexr)、[libultrahdr](https://formulae.brew.sh/formula/libultrahdr) の各ページを参照してください。

Windowsでは、使うエンコーダーを含むFFmpeg・ffprobeと、libmpvのDLL・依存DLLを導入し、アプリから参照できるPATHに配置してください。

### Python環境の準備

```bash
git clone https://github.com/hajimeno-ipoo/SDR2HDR_pro.git
cd SDR2HDR_pro
python -m venv .venv
```

macOS・Linux：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

有効にした仮想環境でインストールします。

```bash
python -m pip install -e ".[ai]"
```

GUIは作業フォルダーの `models/enhancement_model_20260310.pt` を参照します。同梱モデルとフォルダー構成を保ち、リポジトリのフォルダーから起動してください。「更新」はローカルのモデル一覧を読み直す操作で、モデルをダウンロードする操作ではありません。

### 起動

```bash
python -m sdr2hdr.gui
```

インストール後は `sdr2hdr-gui` でも起動できます。

## CLI

CLIは動画AI変換用です。`--model-path` は必須です。GUIのように「変換」と「書き出し」を分けず、処理が完了すると指定先へ保存します。既存の出力先を置き換える前の確認画面はないため、別名を指定してください。

```bash
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 \
  --model-path models/enhancement_model_20260310.pt
```

ProRes出力：

```bash
python -m sdr2hdr.cli input.mp4 output_hdr.mov \
  --encoder prores_422hq --model-path models/enhancement_model_20260310.pt
```

ACEScg連番のZIP：

```bash
python -m sdr2hdr.cli input.mp4 output_aces.zip \
  --encoder openexr_acescg_2_0 --exr-delivery zip \
  --model-path models/enhancement_model_20260310.pt
```

出力先を省略すると、入力名に `_hdr` と形式に合う拡張子を付けます。初期設定のエンコーダーは `libx265` です。

HDR Guidance関連には `--hdr-guidance auto|on|off`、`--luminance-guidance-strength`、`--reconstruction-strength` があります。現在のCLIには、GUIの目標ピーク輝度と彩度に対応する専用引数はありません。

CLIのキャンセル設定はGUIと異なり、既存ファイルがない場合には部分的な動画を残す設定が初期値です。`--discard-partial-output-on-cancel` で保持を無効にできます。すべての引数はヘルプで確認してください。

```bash
python -m sdr2hdr.cli --help
```

### レビュー用の補助コマンド

```bash
sdr2hdr-frames --help
sdr2hdr-compare --help
```

`sdr2hdr-frames` は指定時刻のフレームを抽出し、HDRプレビューPNG・16bit TIFF・線形EXRも作成できます。`sdr2hdr-compare` はSDR/HDRの並列比較PNG、一覧画像、HDR側の線形EXRを作ります。これらのSDR向けプレビューは、アプリ内のHDR表示や納品用の出力とは別の用途です。

## 技術構成と開発

| 項目 | 主な実装・依存関係 |
| --- | --- |
| GUI・一覧管理 | Tkinter / ttk、[gui.py](src/sdr2hdr/gui.py) |
| 変換の実行・保存の確定 | [app.py](src/sdr2hdr/app.py) |
| SDR→HDR処理 | NumPy、OpenCV、PyTorch、[core.py](src/sdr2hdr/core.py)、[ai.py](src/sdr2hdr/ai.py) |
| 出力色変換 | OpenColorIO 2.5.2、[output_color.py](src/sdr2hdr/output_color.py) |
| メディアの読み書き | FFmpeg / ffprobe、PyAV、[io.py](src/sdr2hdr/io.py) |
| 比較表示 | libmpv、macOSではAppKitとネイティブ表示面 |
| GUIの一時出力とコピー | [gui_output.py](src/sdr2hdr/gui_output.py) |
| テスト・学習用スクリプト | [tests](tests/)、[scripts](scripts/) |

Python依存関係とコマンドの定義は [pyproject.toml](pyproject.toml)、macOSのネイティブ拡張のビルド設定は [setup.py](setup.py) にあります。

開発用依存関係とテスト：

```bash
python -m pip install -e ".[ai,dev]"
python -m pytest
```

GUIテストは画面を利用できる環境で実行してください。実ファイルを生成するテストにはFFmpegや対応エンコーダーも必要です。テスト成功と、実際のHDR対応画面での見え方の確認は別です。

## ライセンス

本体は [MIT License](LICENSE) で提供します。外部ライブラリにはそれぞれのライセンスが適用されます。同梱のBT.2100 PQ ICCプロファイルについては [プロファイルの出典と条件](src/sdr2hdr/assets/bt2100-pq.md) を参照してください。
