#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_RS="${1:-$ROOT_DIR/crates/lkh/build.rs}"

if [[ ! -f "$BUILD_RS" ]]; then
  echo "build.rs not found: $BUILD_RS" >&2
  exit 1
fi

extract_const() {
  local name="$1"
  perl -ne 'print "$1\n" if /^\s*const '"${name}"':\s*&str\s*=\s*"([^"]+)";/' "$BUILD_RS"
}

sha256_file() {
  local file="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
    return
  fi

  if command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$file" | awk '{print $NF}'
    return
  fi

  echo "could not find sha256 tool (tried sha256sum, shasum, openssl)" >&2
  exit 1
}

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1 && curl -fsSL "$url" -o "$output"; then
    return
  fi

  if command -v wget >/dev/null 2>&1 && wget -q -O "$output" "$url"; then
    return
  fi

  echo "failed to download archive from $url (tried curl, wget)" >&2
  exit 1
}

LKH_URL="$(extract_const "LKH_HTTP_URL")"
CURRENT_SHA="$(extract_const "LKH_ARCHIVE_SHA256")"

if [[ -z "$LKH_URL" ]]; then
  echo "failed to read LKH_URL from $BUILD_RS" >&2
  exit 1
fi

if [[ -z "$CURRENT_SHA" ]]; then
  echo "failed to read LKH_ARCHIVE_SHA256 from $BUILD_RS" >&2
  exit 1
fi

TMP_ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/lkh-archive.XXXXXX.tgz")"
trap 'rm -f "$TMP_ARCHIVE"' EXIT

echo "Downloading: $LKH_URL"
download_file "$LKH_URL" "$TMP_ARCHIVE"

NEW_SHA="$(sha256_file "$TMP_ARCHIVE" | tr '[:upper:]' '[:lower:]')"

if [[ "$NEW_SHA" == "$CURRENT_SHA" ]]; then
  echo "No change needed: LKH_ARCHIVE_SHA256 is already up to date."
  exit 0
fi

NEW_SHA="$NEW_SHA" perl -i -pe '
  BEGIN { $new = $ENV{"NEW_SHA"}; }
  s/^(\s*const\s+LKH_ARCHIVE_SHA256:\s*&str\s*=\s*")[0-9a-fA-F]{64}(";\s*)$/$1.$new.$2/e
' "$BUILD_RS"

UPDATED_SHA="$(extract_const "LKH_ARCHIVE_SHA256")"
if [[ "$UPDATED_SHA" != "$NEW_SHA" ]]; then
  echo "failed to update LKH_ARCHIVE_SHA256 in $BUILD_RS" >&2
  exit 1
fi

echo "Updated $BUILD_RS"
echo "  old: $CURRENT_SHA"
echo "  new: $NEW_SHA"
