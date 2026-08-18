#!/usr/bin/env bash
# Download COCO-Text dataset: annotations + COCO train2014 images.
#   - annotations: bgshih/cocotext GitHub release (cocotext.v2.zip, ~1.3GB)
#   - images:      COCO train2014 (official images.cocodataset.org 在本机不可达,
#                 使用 pjreddie 镜像 https://data.pjreddie.com/files/train2014.zip, ~13.5GB)
# 断点续传: wget -c;失败自动重试。
set -euo pipefail

BASE_DIR="${DATASET_ROOT:-/home/heqi/dataset}"
DST="$BASE_DIR/cocotext"
mkdir -p "$DST"

dl() {
    local url="$1" out="$2"
    echo "[$(date '+%F %T')] downloading: $url"
    wget -c --tries=10 --timeout=120 --waitretry=15 -O "$out" "$url"
    echo "[$(date '+%F %T')] finished: $out ($(du -h "$out" | cut -f1))"
}

dl "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip" "$DST/cocotext.v2.zip"
dl "https://data.pjreddie.com/files/train2014.zip" "$DST/train2014.zip"

echo "[$(date '+%F %T')] ALL DONE: COCO-Text (annotations + train2014 images)"
