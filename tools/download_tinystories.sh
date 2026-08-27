#!/usr/bin/env bash
set -euo pipefail

# download_tinystories.sh
# Usage: ./download_tinystories.sh <URL> [DEST_DIR]
# Or set environment variable TINYSTORIES_URL and run without args.

url="${1:-${TINYSTORIES_URL:-}}"
dest="${2:-TinyStories}"

if [ -z "$url" ]; then
  echo "Error: no URL provided. Pass the download URL as first arg or set TINYSTORIES_URL."
  echo "Example: ./tools/download_tinystories.sh https://example.com/TinyStories_all_data.tar.gz"
  exit 1
fi

echo "Preparing to download TinyStories to '$dest'"
mkdir -p "$dest"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
archive="$tmpdir/tinystories.tar.gz"

echo "Downloading $url ..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "$archive" "$url"
else
  wget -O "$archive" "$url"
fi

echo "Download finished. Extracting..."
# Try extracting into a temp dir then move contents into dest
tar -xzf "$archive" -C "$tmpdir"

# If the archive contains a single top-level directory, move its contents
first_item_count=$(find "$tmpdir" -mindepth 1 -maxdepth 1 | wc -l)
if [ "$first_item_count" -eq 1 ]; then
  first_item=$(find "$tmpdir" -mindepth 1 -maxdepth 1 | head -n1)
  if [ -d "$first_item" ]; then
    echo "Detected top-level dir $(basename "$first_item"), moving its contents to $dest"
    shopt -s dotglob
    mv "$first_item"/* "$dest" || true
    shopt -u dotglob
  else
    mv "$tmpdir"/* "$dest" || true
  fi
else
  mv "$tmpdir"/* "$dest" || true
fi

echo "Extraction complete. Listing $dest:"
ls -la "$dest" | sed -n '1,200p'

echo "TinyStories setup complete."
