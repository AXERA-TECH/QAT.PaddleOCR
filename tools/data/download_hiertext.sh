#!/usr/bin/env bash
# Download HierText dataset: images (Open Images subset, public S3) + annotations.
#   - images:      s3://open-images-dataset/ocr/{train,validation,test}.tgz
#                  (https 直连 open-images-dataset.s3.amazonaws.com, 已验证可达;
#                  train 2.8GB / validation 0.58GB / test 0.54GB)
#   - annotations: github raw gt/{train,validation,test}.jsonl.gz (word/line/paragraph)
set -euo pipefail

BASE_DIR="${DATASET_ROOT:-/home/heqi/dataset}"
DST="$BASE_DIR/hiertext"
mkdir -p "$DST/images" "$DST/annotations"

dl() {
    local url="$1" out="$2"
    echo "[$(date '+%F %T')] downloading: $url"
    wget -c --tries=10 --timeout=120 --waitretry=15 -O "$out" "$url"
    echo "[$(date '+%F %T')] finished: $out ($(du -h "$out" | cut -f1))"
}

for split in train validation test; do
    dl "https://open-images-dataset.s3.amazonaws.com/ocr/${split}.tgz" "$DST/images/${split}.tgz"
done

for split in train validation test; do
    dl "https://raw.githubusercontent.com/google-research-datasets/hiertext/main/gt/${split}.jsonl.gz" \
       "$DST/annotations/${split}.jsonl.gz"
done

echo "[$(date '+%F %T')] ALL DONE: HierText (images + annotations)"
