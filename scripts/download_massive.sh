#!/usr/bin/env bash
# Fetch MASSIVE 1.1 (Amazon, CC BY 4.0) and keep only the en-US / hi-IN / bn-BD files.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
curl -sSL -o "$tmp/massive.tar.gz" https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf "$tmp/massive.tar.gz" -C "$tmp" 1.1/LICENSE 1.1/data/en-US.jsonl 1.1/data/hi-IN.jsonl 1.1/data/bn-BD.jsonl
mkdir -p data/raw
cp "$tmp"/1.1/data/*.jsonl data/raw/
cp "$tmp/1.1/LICENSE" data/raw/MASSIVE_LICENSE
rm -rf "$tmp"
wc -l data/raw/*.jsonl
