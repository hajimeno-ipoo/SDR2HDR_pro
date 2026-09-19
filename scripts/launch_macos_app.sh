#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python="$project_root/.venv/bin/python"
icon="$project_root/design/app-icon/SDR2HDRPro.icns"
app="$project_root/build/SDR2HDR Pro.app"
contents="$app/Contents"

if [ ! -x "$python" ]; then
    echo "Python環境がありません: $python" >&2
    exit 1
fi
if [ ! -f "$icon" ]; then
    echo "アプリアイコンがありません: $icon" >&2
    exit 1
fi
python_app_binary=$("$python" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python")')
if [ ! -x "$python_app_binary" ]; then
    echo "macOS用Python実行ファイルがありません: $python_app_binary" >&2
    exit 1
fi

mkdir -p "$contents/MacOS" "$contents/Resources"
rm -f "$contents/Info.plist" "$contents/MacOS/launch" "$contents/MacOS/SDR2HDRPro" \
    "$contents/Resources/launch.py" "$contents/Resources/SDR2HDRPro.icns"
cat > "$contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>ja</string>
    <key>CFBundleDisplayName</key><string>SDR2HDR Pro</string>
    <key>CFBundleExecutable</key><string>SDR2HDRPro</string>
    <key>CFBundleIconFile</key><string>SDR2HDRPro.icns</string>
    <key>CFBundleIdentifier</key><string>local.sdr2hdr.pro</string>
    <key>CFBundleName</key><string>SDR2HDR Pro</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
</dict>
</plist>
PLIST

cp "$python_app_binary" "$contents/MacOS/SDR2HDRPro"
chmod +x "$contents/MacOS/SDR2HDRPro"

cat > "$contents/Resources/launch.py" <<'LAUNCH'
import os
from pathlib import Path
import site
import sys

project_root = Path(__file__).resolve().parents[4]
venv_site = project_root / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if not venv_site.is_dir():
    raise SystemExit(f"Python環境がありません: {venv_site}")
site.addsitedir(str(venv_site))
sys.path.insert(0, str(venv_site))
sys.path.insert(0, str(project_root / "src"))
os.environ["PYTHONPATH"] = os.pathsep.join((str(project_root / "src"), str(venv_site), os.environ.get("PYTHONPATH", "")))
os.chdir(project_root)

from sdr2hdr.gui import main

raise SystemExit(main())
LAUNCH

cp "$icon" "$contents/Resources/SDR2HDRPro.icns"

open "$app" --args "$contents/Resources/launch.py"
