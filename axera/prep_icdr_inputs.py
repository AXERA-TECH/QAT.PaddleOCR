"""Preprocess ICDR test word images into 100-image parts."""
import argparse
import cv2, numpy as np, os, json, math, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("data_dir", help="directory containing rec_gt_test.txt and images")
parser.add_argument("out_dir", help="output directory for part_*.npz")
args = parser.parse_args()
data_dir = args.data_dir
label_file = os.path.join(data_dir, "rec_gt_test.txt")
shape = (3, 48, 320)
lines = [l.strip() for l in open(label_file) if l.strip()]
imgs, texts = [], []
for line in lines:
    img_path, text = line.split("\t", 1)
    img = cv2.imread(os.path.join(data_dir, img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {img_path}")
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
out_dir = args.out_dir
shutil.rmtree(out_dir, ignore_errors=True)
os.makedirs(out_dir)
for i in range(0, len(arr), 100):
    np.savez_compressed(f"{out_dir}/part_{i // 100:03d}.npz", data=arr[i:i + 100])
json.dump(texts, open(os.path.join(os.path.dirname(out_dir), "icdr_texts.json"), "w"))
print(f"prepared {len(imgs)} inputs -> {out_dir} ({ (len(arr)+99)//100 } parts)")
