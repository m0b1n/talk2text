#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
DEPLOY_DIR="$BUILD_DIR/deploy"
PKG_DIR="$BUILD_DIR/pkgroot"
RAW_DEPLOY_DIR="$ROOT_DIR/deployment"
PKG_NAME="talk2text"
APP_ID="io.github.m0b1n.Talk2Text"
ICON_NAME="$APP_ID"
DEPLOY_SPEC="$ROOT_DIR/pysidedeploy.spec"
VENV_PYTHON="${VENV_PYTHON:-$ROOT_DIR/.venv/bin/python}"
PYSIDE_DEPLOY_BIN="${PYSIDE_DEPLOY_BIN:-$ROOT_DIR/.venv/bin/pyside6-deploy}"
ARCH="${DEB_ARCH:-$(dpkg --print-architecture)}"
VERSION="$("$VENV_PYTHON" -c 'import tomllib, pathlib; data = tomllib.loads(pathlib.Path("pyproject.toml").read_text()); print(data["project"]["version"])')"
DEB_VERSION="${DEB_VERSION:-${VERSION}-1}"
OUTPUT_DEB="$ROOT_DIR/${PKG_NAME}_${DEB_VERSION}_${ARCH}.deb"

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf 'Missing required file: %s\n' "$path" >&2
    exit 1
  fi
}

require_file "$VENV_PYTHON"
require_file "$PYSIDE_DEPLOY_BIN"
require_file "$DEPLOY_SPEC"
require_file "$ROOT_DIR/packaging/main.py"
require_file "$ROOT_DIR/packaging/${APP_ID}.desktop"
require_file "$ROOT_DIR/packaging/${ICON_NAME}.svg"

rm -rf "$DEPLOY_DIR" "$PKG_DIR" "$RAW_DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR" "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/opt/talk2text"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/applications"
mkdir -p "$PKG_DIR/usr/share/icons/hicolor/scalable/apps"

"$PYSIDE_DEPLOY_BIN" \
  -c "$DEPLOY_SPEC" \
  --mode standalone \
  --force \
  --keep-deployment-files

if [[ ! -d "$RAW_DEPLOY_DIR" ]]; then
  printf 'pyside6-deploy did not create the expected deployment directory.\n' >&2
  exit 1
fi

rm -rf "$DEPLOY_DIR"
mv "$RAW_DEPLOY_DIR" "$DEPLOY_DIR"

APP_BIN_RELATIVE_PATH="$("$VENV_PYTHON" - <<'PY' "$DEPLOY_DIR"
from pathlib import Path
import sys

deploy_dir = Path(sys.argv[1])
bin_files = sorted(deploy_dir.glob("*.dist/*.bin"))
if len(bin_files) != 1:
    raise SystemExit(f"Expected exactly one .bin file in {deploy_dir}, found {len(bin_files)}")
print(bin_files[0].relative_to(deploy_dir).as_posix())
PY
)"

cp -a "$DEPLOY_DIR/." "$PKG_DIR/opt/talk2text/"
cp "$ROOT_DIR/packaging/${APP_ID}.desktop" "$PKG_DIR/usr/share/applications/${APP_ID}.desktop"
cp "$ROOT_DIR/packaging/${ICON_NAME}.svg" "$PKG_DIR/usr/share/icons/hicolor/scalable/apps/${ICON_NAME}.svg"

cat >"$PKG_DIR/usr/bin/talk2text" <<EOF
#!/bin/sh
exec /opt/talk2text/${APP_BIN_RELATIVE_PATH} "\$@"
EOF
chmod 0755 "$PKG_DIR/usr/bin/talk2text"

cat >"$PKG_DIR/DEBIAN/control" <<EOF
Package: ${PKG_NAME}
Version: ${DEB_VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: m0b1n <ghahramanpour.mobin@gmail.com>
Depends: libc6, libgl1, libglib2.0-0, libnss3, libstdc++6
Description: Local Ubuntu speech-to-text desktop app
 Talk2Text records microphone audio, transcribes it locally with
 faster-whisper, and can optionally enhance the transcript with a
 local Ollama model.
EOF

dpkg-deb --root-owner-group --build "$PKG_DIR" "$OUTPUT_DEB"
printf 'Built %s\n' "$OUTPUT_DEB"
