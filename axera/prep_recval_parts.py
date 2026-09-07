"""Preprocess rec_val into memory-bounded parts for board evaluation."""
import argparse
import cv2, numpy as np, os, json, math, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data_dir", help="directory containing rec_val.txt and val/")
parser.add_argument("out_dir", help="output directory for part_*.npz")
args = parser.parse_args()
data_dir = args.data_dir
out_dir = args.out_dir
shape = (3, 48, 320)
lines = [l.strip() for l in open(os.path.join(data_dir, "rec_val.txt")) if l.strip()]
imgs, texts = [], []
for line in lines:
    img_path, text = line.split("\t", 1)
    name = os.path.basename(img_path)
    full = os.path.join(data_dir, "val", name)
    img = cv2.imread(full, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {full}")
    h, w = img.shape[:2]
    rw = min(shape[2], int(math.ceil(shape[1] * w / h)))
    resized = cv2.resize(img, (rw, shape[1])).astype(np.float32)
    resized = resized.transpose(2, 0, 1) / 255.0
    resized = (resized - 0.5) / 0.5
    out = np.zeros(shape, dtype=np.float32)
    out[:, :, :rw] = resized
    imgs.append(out)
    texts.append(text)
arr = np.stack(imgs)
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir)
step = 500
for i in range(0, len(arr), step):
    np.savez_compressed(f"{out_dir}/part_{i // step:04d}.npz", data=arr[i:i + step])
json.dump(texts, open(os.path.join(out_dir, "..", "recval_texts.json"), "w"))
print(f"prepared {len(imgs)} -> {out_dir} ({ (len(arr)+step-1)//step } parts)")
