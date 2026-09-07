# SDR2HDR_pro

SDR動画・画像をHDRへ変換するPythonアプリです。GUIでの一括変換と、CLIでの動画変換に対応しています。

元作者・元プロジェクト：[jpneagle / sdr2hdr](https://github.com/jpneagle/sdr2hdr)

## 主な機能

- 動画・画像のHDR変換
- 複数ファイルを順番に処理する変換キュー
- HDR10（HEVC）、ProRes、OpenEXR連番の出力
- 明るさや色の調整用プリセット

利用できる出力形式は、OSとFFmpegの構成によって異なります。

## 必要な環境

- Python 3.11以上
- FFmpegとffprobe（コマンドとして実行できること）
- Tkinter（GUIを使う場合）

PyTorchなどのPython依存関係は、次のインストール手順で導入します。

## インストール

```bash
git clone https://github.com/hajimeno-ipoo/SDR2HDR_pro.git
cd SDR2HDR_pro
python -m venv .venv
```

macOS・Linuxでは、次のコマンドで仮想環境を有効にします。

```bash
source .venv/bin/activate
```

WindowsのPowerShellでは、次を実行します。

```powershell
.\.venv\Scripts\Activate.ps1
```

有効にした環境で依存関係をインストールします。

```bash
python -m pip install -e ".[ai]"
```

変換に使うモデルは `models/enhancement_model_20260310.pt` として同梱しています。フォルダ構成を変えずに使用してください。

## 使い方

### GUI

リポジトリのフォルダで起動します。

```bash
python -m sdr2hdr.gui
```

動画または画像のタブで入力ファイル、出力先、プリセット、出力形式を指定し、キューに追加して変換を開始します。

### CLI

```bash
python -m sdr2hdr.cli input.mp4 output_hdr.mp4 --model-path models/enhancement_model_20260310.pt
```

その他の指定項目はヘルプで確認できます。

```bash
python -m sdr2hdr.cli --help
```

## 出力時の注意

- HDRの見え方を確認するには、HDR対応の表示環境が必要です。
- 通常のOpenEXR出力はBT.2020/PQです。AP1線形出力は表示基準の線形値で、撮影時の露光量を復元するものではありません。編集ソフトでは出力に合った色空間を指定してください。
- AP1線形OpenEXR出力には、OpenEXRの `exrstdattr` コマンドが別途必要です。
- ProRes 4444 XQの12-bit出力には、対応するmacOSのVideoToolboxとFFmpegが必要です。
- 変換による画質向上や、白飛びで失われた細部の復元を保証するものではありません。

## ライセンス

[MIT License](LICENSE)
