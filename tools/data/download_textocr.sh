#!/usr/bin/env bash
# Download TextOCR dataset: images (TextVQA/Open Images subset) + annotations.
#   - images:      https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip (~7GB)
#   - annotations: https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_{train,val,test}.json
set -euo pipefail

BASE_DIR="${DATASET_ROOT:-/home/heqi/dataset}"
DST="$BASE_DIR/textocr"
mkdir -p "$DST/images" "$DST/annotations"

dl() {
    local url="$1" out="$2"
    echo "[$(date '+%F %T')] downloading: $url"
    wget -c --tries=10 --timeout=120 --waitretry=15 -O "$out" "$url"
    echo "[$(date '+%F %T')] finished: $out ($(du -h "$out" | cut -f1))"
}

dl "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" "$DST/images/train_val_images.zip"

for split in train val test; do
    dl "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_${split}.json" \
       "$DST/annotations/TextOCR_0.1_${split}.json"
done

echo "[$(date '+%F %T')] ALL DONE: TextOCR (images + annotations)"
