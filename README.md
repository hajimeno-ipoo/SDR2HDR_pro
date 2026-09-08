# SDR2HDR_pro

SDR動画・画像をHDRへ変換するPythonアプリです。GUIでの一括変換と、CLIでの動画変換に対応しています。

元作者・元プロジェクト：[jpneagle / sdr2hdr](https://github.com/jpneagle/sdr2hdr)

## 主な機能

- 動画・画像のHDR変換
- 複数ファイルを順番に処理する変換キュー
- HDR10・HLG（HEVC 10bit）、ProRes、OpenEXR連番の出力
- ACES 1.3／2.0の公式変換によるACEScg OpenEXR出力
- 明るさや色の調整用プリセット

利用できる出力形式は、OSとFFmpegの構成によって異なります。

## 必要な環境

- Python 3.11以上
- FFmpegとffprobe（コマンドとして実行できること）
- Tkinter（GUIを使う場合）

OpenColorIO 2.5.2、PyTorchなどのPython依存関係は、次のインストール手順で導入します。

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

出力形式は `--encoder` で選択します。出力先を省略すると、形式に合う拡張子を付けます。ProResはMOV、OpenEXRは連番画像と音声（ある場合）をまとめたZIPへ保存します。EXR系は初期設定でZIPを保存します。GUIの「保存方法」またはCLIの `--exr-delivery video` でHDR動画（ProRes 4444 / MOV）を選べます。動画の場合は出力先を `.mov` にしてください。ACEScgの動画保存では、選択したACES版の公式出力変換でBT.2020/PQへ変換します。ACEScgデータを保持する場合はZIPを選んでください。

| 出力 | 指定値 |
|---|---|
| HLG / HEVC 10bit | `hevc_hlg` |
| ProRes 422 HQ / 10bit | `prores_422hq` |
| ProRes 4444 / 10bit入力 | `prores_4444` |
| ProRes 4444 XQ / 12bit | `prores_4444_xq` |
| ACEScg / ACES 1.3 | `openexr_acescg_1_3` |
| ACEScg / ACES 2.0 | `openexr_acescg_2_0` |

HLGとACEScgは、GUIの「動画 AI」タブでも選択できます。

その他の指定項目はヘルプで確認できます。

```bash
python -m sdr2hdr.cli --help
```

## 出力時の注意

- HDRの見え方を確認するには、HDR対応の表示環境が必要です。
- 通常のOpenEXR出力はBT.2020/PQです。AP1線形出力は表示基準の線形値で、撮影時の露光量を復元するものではありません。編集ソフトでは出力に合った色空間を指定してください。
- AP1線形とACEScgのOpenEXR出力には、OpenEXRの `exrstdattr` コマンドが別途必要です。
- ACEScgは、HDR変換後の映像に公式ACES出力変換の逆変換を適用します。撮影時に失われた情報を復元する処理ではありません。ACESの設定と変換名をEXRへ記録します。従来の表示基準AP1出力は別の選択肢として残しています。
- ACEScg出力は、[OpenColorIOの公式ACES設定](https://opencolorio.readthedocs.io/en/v2.5.2/configurations/aces_cg.html)に含まれる1000 nit・Rec.2020の出力変換を逆方向に適用します。公式変換の切り詰めや色域処理もそのまま使用します。往復後の数値差だけを不具合とは扱わず、公式変換との整合性とEXRの保存精度を確認します。専用の表示変換は不要です。
- HLGとACEScgは基準ピーク1000 nitの変換です。出力上限は1000 nit以下を使用してください。既存プリセットはすべてこの範囲です。
- ProResは映像内部とMOVの両方にBT.2020/PQの色情報を記録します。記録処理では映像や音声を再圧縮しません。
- ProRes 4444 XQの12-bit出力には、対応するmacOSのVideoToolboxとFFmpegが必要です。
- 変換による画質向上や、白飛びで失われた細部の復元を保証するものではありません。

## ライセンス

[MIT License](LICENSE)

### HDR JPEGとPNGの画像出力

画像AIと画像Logの出力形式にJPEGとPNGを追加しました。JPEGはlibultrahdrの公式エンコーダーで、変換後のBT.2020/PQ画素を10bit RGB入力へ量子化し、SDR画像とHDR復元用ゲインマップを含むUltra HDR JPEGとして保存します。非可逆圧縮のため画素の完全一致は保証しません。拡張子は `.jpg` または `.jpeg` です。`ultrahdr_app` がPATHに必要です（検証版1.4.0、macOSでは `brew install libultrahdr`）。未導入時は変換開始前にエラーを表示します。

PNGは16bit RGBをロスレス保存し、PNG第3版のcICPチャンク（9,16,0,1）でBT.2020、PQ、RGB、フルレンジを明示します。JPEGとPNGのHDR表示には対応する閲覧アプリとHDR表示環境が必要です。Ultra HDR非対応のJPEG閲覧アプリではSDR画像を表示します。

### 入力の複数選択とキュー登録

入力の参照から1件または複数件を選択します。「ファイルを追加」は削除しました。1件の場合は従来どおり出力ファイル名を指定し、複数件の場合は共通の保存先フォルダーを指定します。「キューに追加」で現在の変換設定を各ファイルへ適用し、1ファイルを1行として個別登録します。キューが空の状態で開始した場合も、選択した全件を登録します。複数件の出力名は各入力名を基に生成し、既存ファイルや登録済みの出力名と重なる場合は番号を付けます。形式変更時も保存先フォルダーは維持します。
