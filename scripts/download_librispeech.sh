#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/raw/librispeech}"
BASE_URL="https://openslr.org/resources/12"
SPLITS=("train-clean-100" "dev-clean" "test-clean")

mkdir -p "${ROOT}"

for split in "${SPLITS[@]}"; do
  archive="${split}.tar.gz"
  url="${BASE_URL}/${archive}"
  target="${ROOT}/${archive}"

  echo "==> Downloading ${archive}"
  if [ ! -f "${target}" ]; then
    wget -c "${url}" -O "${target}"
  else
    echo "Archive already exists: ${target}"
  fi
  echo "==> Extracting ${archive}"
  tar -xzf "${target}" -C "${ROOT}"
done

echo "LibriSpeech splits available under ${ROOT}"
