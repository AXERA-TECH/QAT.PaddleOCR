"""Board-side rec eval: AxEngineExecutionProvider + CTC decode, low memory (parts)."""
import argparse
import json
import sys
import numpy as np
try:
    import axengine as ort
    print("Using AXEngine (NPU)", flush=True)
except ImportError:
    import onnxruntime as ort
    print("Using onnxruntime CPU", flush=True)
    PROVIDERS = ["CPUExecutionProvider"]
else:
    PROVIDERS = ["AxEngineExecutionProvider"]

def load_dictionary(path, use_space_char=False):
    characters = [l.rstrip("\r\n") for l in open(path, encoding="utf-8")]
    if use_space_char:
        characters.append(" ")
    return characters

def decode_indices(indices, characters):
    ignored = {0}
    class_count = len(characters) + 1
    texts = []
    for sequence in indices:
        decoded, previous = [], None
        for raw in sequence:
            index = int(raw)
            dup = previous == index
            previous = index
            if dup or index in ignored:
                continue
            if not 0 <= index < class_count:
                raise ValueError(f"class {index} out of range")
            decoded.append(characters[index - 1])
        texts.append("".join(decoded))
    return texts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axmodel", required=True)
    parser.add_argument("--texts", required=True, help="JSON list aligned with input parts")
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-space-char", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    import glob
    texts = json.load(open(args.texts, encoding="utf-8"))
    characters = load_dictionary(args.dictionary, args.use_space_char)
    session = ort.InferenceSession(
        args.axmodel, sess_options=ort.SessionOptions(), providers=PROVIDERS
    )
    input_name = session.get_inputs()[0].name
    preds = []
    total = 0
    for part in sorted(glob.glob(args.parts_dir + "/part_*.npz")):
        inputs = np.load(part)["data"]
        total += inputs.shape[0]
        for start in range(0, inputs.shape[0], args.batch_size):
            chunk = inputs[start:start + args.batch_size]
            logits = session.run(None, {input_name: chunk.astype(np.float32)})[0]
            preds.extend(decode_indices(np.argmax(logits, axis=-1), characters))
        del inputs
    if total != len(texts):
        raise ValueError(f"prediction count {total} != reference text count {len(texts)}")
    matches = sum(1 for p, t in zip(preds, texts) if p == t)
    import difflib
    neds = [difflib.SequenceMatcher(None, p, t).ratio() for p, t in zip(preds, texts)]
    print(f"acc={matches/total:.6f} norm_edit_sim={sum(neds)/total:.6f} samples={total}", flush=True)

if __name__ == "__main__":
    main()
