#!/usr/bin/env bash
set -euo pipefail

echo "[lkh] formatting check"
cargo fmt --all --check

echo "[lkh] tests (no default features)"
cargo test -p lkh --no-default-features

echo "[lkh] clippy (no default features)"
cargo clippy -p lkh --no-default-features --all-targets -- -D warnings

echo "[lkh] package manifest/file check"
cargo package -p lkh --allow-dirty --list >/dev/null

if [[ "${LKH_RELEASE_FULL_PACKAGE:-0}" == "1" ]]; then
  echo "[lkh] tests (default features)"
  cargo test -p lkh

  echo "[lkh] clippy (default features)"
  cargo clippy -p lkh --all-targets -- -D warnings

  echo "[lkh] full cargo package verification"
  cargo package -p lkh --allow-dirty
fi

echo "[lkh] release checks passed"
