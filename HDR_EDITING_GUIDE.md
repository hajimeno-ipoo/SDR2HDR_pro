# HDR動画を編集ソフトで扱うためのカラー設定

SDR2HDR Proから書き出したHDR動画について、参照会話「HDR書き出し設定比較」の内容を整理した資料です。一覧表、ソフト別の設定、HLGの場合の違い、Mac側の表示設定をまとめています。

元の会話は2026年9月時点の設定を扱っています。ソフト名・バージョン・設定名はその会話に基づきます。今回の文書化では、公式資料の再調査や各編集ソフトでの実機検証は行っていません。

[READMEへ戻る](README.md)

## 対象となる出力

通常のHEVC/PQとProRes出力は、**Rec.2020原色とPQ（ST 2084）を組み合わせたHDR素材**として扱います。編集ソフトでは、これに対応する名前が `Rec.2100 PQ`、`Rec.2100 ST2084`、`Rec.2020 PQ` などになっています。

| SDR2HDR Proの出力 | 編集ソフトでの基本的な扱い |
| --- | --- |
| HEVC / PQ | Rec.2020原色 + PQ（ST 2084） |
| ProRes 422 HQ | Rec.2020原色 + PQ（ST 2084） |
| ProRes 4444 | Rec.2020原色 + PQ（ST 2084） |
| ProRes 4444 XQ | Rec.2020原色 + PQ（ST 2084） |
| HEVC / HLG | Rec.2020原色 + HLG（ARIB STD-B67）。PQとは別設定 |

保存形式やビット深度が違っても、上記のPQ出力の基本カラー設定は共通です。目標ピーク輝度を1000 nitから600 nitなどへ変えても、PQ素材の色空間を別のものへ変更する必要はありません。

この資料はHDRのまま編集・表示・再出力する場合を対象にしています。SDR納品への変換、静止画、OpenEXR連番のカラー設定は対象外です。

## PQ動画の設定一覧

| 編集ソフト | 素材の色空間 | プロジェクト／タイムライン | HDR表示 | HDRのまま出力 |
| --- | --- | --- | --- | --- |
| DaVinci Resolve 21 | `Rec.2100 ST2084` | `DaVinci YRGB Color Managed`。Input：`Rec.2100 ST2084`、Timeline：`DaVinci Wide Gamut / Intermediate`、Output：`Rec.2100 ST2084` | HDR対応表示環境を使用 | `Rec.2100 ST2084` |
| Final Cut Pro | 自動認識。PQの色タグは `9-16-9` | ライブラリ：`Wide Gamut HDR`。プロジェクト：`Wide Gamut HDR - Rec. 2020 PQ` | 対応XDRディスプレイでは `HDR Video (P3-ST 2084)` | `Wide Gamut HDR - Rec. 2020 PQ` |
| Adobe Premiere Pro | `Rec.2100 PQ` | 単純編集：`Direct PQ (HDR)`。グレーディング：`Wide Gamut (Tone Mapped)`、Output：`Rec.2100 PQ` | `Display Color Management` と `Extended Dynamic Range Monitoring` をON | `Rec.2100 PQ` |
| After Effects | `Rec.2100 PQ` | Color Engine：`Adobe Color Managed`、32 bpc、Working Color Space：`Rec.2100 PQ` | HDR表示を有効化 | `Rec.2100 PQ` |
| VEGAS Pro 23 | HDR10/PQ素材として読み込み | `Project Properties` → `HDR Mode = HDR10` | `HDR Preview` | HDR対応の出力設定 |
| EDIUS 11 | `BT.2020/BT.2100 PQ` | Project Color Space：`BT.2020/BT.2100 PQ`、Video Bit Depth：`10bit` | HDR対応プレビュー | PQ/HDR対応形式 |
| CapCut Desktop | 元の会話ではHDR10の読み込み・編集に対応と整理 | 入力・作業・出力のPQ色空間を明示的に指定する詳細設定は、元の会話では確認できていない | HDR対応環境 | HDR出力を選択。詳細なカラー管理条件は未確認 |

色空間の指定だけでHDR表示が成立するわけではありません。対応画面、OS側のHDR設定、編集ソフトの表示設定も必要です。

## ソフト別の設定と注意

### DaVinci Resolve 21

参照会話で示された基準設定です。

| 設定 | 値 |
| --- | --- |
| Color science | `DaVinci YRGB Color Managed` |
| Automatic color management | OFF |
| Input Color Space | `Rec.2100 ST2084` |
| Timeline Color Space | `DaVinci Wide Gamut / Intermediate` |
| Output Color Space | `Rec.2100 ST2084` |
| Data Levels | `Auto` |
| HDR mastering | アプリで指定したピーク輝度を基準に設定 |
| ピーク輝度が1000 nitの場合の例 | `1000 nit` |

Data Levelsはまず `Auto` とし、素材を根拠なく `Full` に固定しないでください。色空間とデータ範囲は別の設定です。

読み込み時に素材の色空間を誤認した場合だけ、メディアプールでクリップを右クリックし、Input Color Spaceを `Rec.2100 ST2084` に指定します。正常に認識している場合は上書きする必要はありません。

### Final Cut Pro

| 対象 | 設定 |
| --- | --- |
| ライブラリ | `Wide Gamut HDR` |
| プロジェクト | `Wide Gamut HDR - Rec. 2020 PQ` |
| PQ素材の色タグ | `9-16-9` |

`9-16-9` は、Rec.2020原色、PQ、BT.2020 non-constant luminanceの組み合わせです。色タグの確認と、プロジェクトをPQに設定することを分けて確認します。

### Adobe Premiere Pro

| 編集目的 | 設定 |
| --- | --- |
| すでにHDR化したPQ素材を、基本的な見た目を保って編集 | `Direct PQ (HDR)` |
| 露出・ハイライト・色調を積極的に編集 | `Wide Gamut (Tone Mapped)`、Output Color Space：`Rec.2100 PQ` |

入力素材は `Rec.2100 PQ` として扱い、表示側では `Display Color Management` と `Extended Dynamic Range Monitoring` を有効にします。

### After Effects

| 設定 | 値 |
| --- | --- |
| Color Engine | `Adobe Color Managed` |
| Bit Depth | `32 bpc` |
| Working Color Space | `Rec.2100 PQ` |
| Output Color Space | `Rec.2100 PQ` |

素材の色空間を誤認した場合だけ、`Interpret Footage` から `Media Color Space` を上書きします。

### VEGAS Pro 23

`Project Properties` → `HDR Mode` → `HDR10` を選びます。

参照会話では、この選択により `32-bit floating point (full range)`、`Rec 2020 ST2084 1000 nits (ACES)`、`HDR Preview` などが設定されると説明されています。

ここでの `full range` はVEGAS内部の32bit演算モードの名称です。入力動画を強制的にFull Range素材として解釈する指定とは区別してください。

### EDIUS 11

| 設定 | 値 |
| --- | --- |
| Color Space | `BT.2020/BT.2100 PQ` |
| Video Bit Depth | `10bit` |

入力、プロジェクト、出力をPQ/HDRに対応した設定で扱います。

### CapCut Desktop

元の会話ではHDR10素材の読み込み・編集対応が紹介されています。ただし、入力PQ・作業色空間・出力PQを個別に指定する詳細な設定は確認されていません。

このため、CapCutについては上表だけで正確なカラー管理が確認済みとは扱いません。カラー設定を明示して確認したい場合、元の会話ではResolve、Final Cut Pro、Premiereを候補に挙げています。

## HLGで書き出した場合

HEVC/HLGで書き出した素材には、PQの設定をそのまま使わず、HLGに対応する設定を選びます。

| 編集ソフト | PQの場合 | HLGの場合 |
| --- | --- | --- |
| DaVinci Resolve | `Rec.2100 ST2084` | `Rec.2100 HLG` |
| Final Cut Pro | `Wide Gamut HDR - Rec. 2020 PQ` | `Wide Gamut HDR - Rec. 2020 HLG` |
| Adobe Premiere Pro | `Direct PQ (HDR)` | `Direct HLG (HDR)` |
| After Effects | `Rec.2100 PQ` | `Rec.2100 HLG` |
| VEGAS Pro | `HDR10` | `HLG` |
| EDIUS | `BT.2020/BT.2100 PQ` | `BT.2020/BT.2100 HLG` |

Final Cut Proで確認する色タグも、PQの `9-16-9` からHLGの `9-18-9` に変わります。

## Mac側のHDR表示設定

参照会話では、Liquid Retina XDRなど対応ディスプレイでのPQ映像制作向け設定として、次が紹介されています。

**システム設定 → ディスプレイ → プリセット → `HDR Video (P3-ST 2084)`**

この表示モードは、参照会話ではST 2084のHDR映像制作と、全画面持続1000 nitまでのBT.2100ワークフロー向けと説明されています。利用できるプリセットはディスプレイに依存します。

ディスプレイのプリセット名にP3とあっても、元のPQ動画の入力色空間をP3へ変更するという意味ではありません。素材の解釈と画面の表示条件を分けて設定します。

## 表示がおかしいときの確認点

- PQ素材を `Rec.709` や `Rec.709 Scene` として誤解釈していないか。
- PQ素材の数値を、適切な色変換なしでRec.709の数値として扱っていないか。SDR納品を目的に正しく変換する場合とは別です。
- 入力色空間、作業色空間、出力色空間が、意図したHDR処理になっているか。
- Data Levelsを根拠なく `Full` に固定していないか。
- HDR対応画面とOS・アプリのHDR表示設定を使用しているか。
- HLGで書き出した素材にPQ設定を使っていないか。

これらは確認項目です。特定のファイルが白っぽく見える原因を、この資料だけで断定するものではありません。

## 出典と確認範囲

- 元の会話：[HDR書き出し設定比較](chatgpt-conversation://6aa5673f-41ec-83ee-83fb-7a1f86db2f3b)
- アプリの保存仕様：[README「書き出し仕様」](README.md#書き出し仕様)
- 文書化日：2026年9月13日

元の会話には公式資料への引用記号がありますが、取得した会話本文には参照先URLが含まれていませんでした。そのため、利用できない引用記号を転載せず、元の会話を出典として示しています。各編集ソフトでの実機確認や公式資料の再検証を行った資料ではありません。
