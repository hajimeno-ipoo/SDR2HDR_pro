# SDR2HDR Pro アプリアイコン

元画像の女の子を中心に、顔・上半身・ヘッドホン・カチンコを残したアイコン。ピンク・緑・黒・白の配色を引き継ぐ。元画像とアプリのヘッダー画像は変更していない。

## 素材と出力

- 入力：`/Users/apple/Desktop/exec-35703012-76b8-4d6b-ad48-cc26c90748fe.png`
- 作成方法：組み込みImageGenで構図を変更。最初の出力は透明部分が市松模様として描かれたため不採用とし、正方形の不透明画像へ修正。
- 最終生成画像：`/Users/apple/.codex/generated_images/01a091aa-7fb2-7eb0-ac75-6c84f45608de/exec-123c8f36-d98d-4149-8934-0a3d36c5f20d.png`
- 編集元：`SDR2HDRPro.icon`。生成画像をsipsで1024×1024へ縮小して格納。
- アプリが使う出力：`../../src/sdr2hdr/assets/app-icon.png`。Icon Composer 1.6のmacOS Defaultで書き出した1024×1024、透過付きPNG。
- アイコンの設定：Tk標準の`iconphoto`を起動時に呼ぶ。Python.appの設定ファイルや他のアプリは変更しない。

## 最終生成プロンプト

入力1は最初に生成したアイコン、入力2は利用者の元画像。

> Correct image 1 into flat full-bleed square icon artwork. Keep the same girl, face, headphones, shirt, pose and clapperboard; girl remains centered and large. Remove ALL gray checkerboard pixels, ALL outer margins, ALL rounded tile edges, ALL icon shadow and highlights. Fill the whole square canvas to all four straight edges with one uniform pastel pink sampled from image 2 (the original banner), less saturated than image 1. This must be an OPAQUE full-bleed square illustration, no transparent areas, no fake transparency pattern, no tile or rounded corners; Apple's icon renderer will add masking afterward. Preserve original image 2 sage green/black/white color scheme and girl's identity. Entire hair and clapperboard inside central 85% of canvas with pink breathing room, torso naturally continues off bottom edge. No text, no new objects. Output one 1024x1024 square PNG.

## 書き出し

Icon Composerスキル付属の検証で`SDR2HDRPro.icon`の構造と素材参照を確認した後、次の形で書き出す。各パスには絶対パスを使う。

```sh
ictool /absolute/path/SDR2HDRPro.icon --export-image --output-file /absolute/path/app-icon.png --platform macOS --rendition Default --width 1024 --height 1024 --scale 1
```

根拠：[Tk 9のiconphoto仕様](https://www.tcl-lang.org/man/tcl9.0/TkCmd/wm.html#M71)。macOSでは最初の画像がDockやダイアログなどのアプリアイコンに使われる。配布用アプリへの梱包やFinder上のPython.appのアイコン変更は今回の対象外。
