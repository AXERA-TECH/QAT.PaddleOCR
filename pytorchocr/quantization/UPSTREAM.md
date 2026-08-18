# QAT.axera Vendor Record

The files in this directory are copied from:

```text
repository: https://github.com/AXERA-TECH/QAT.axera
commit:     4603b160bad6b212551721ec3cd3b75895b17de3
license:    BSD-3-Clause
```

Vendored source files:

```text
utils/ax_quantizer.py
utils/ax_quantizer_utils.py
utils/quantized_decomposed_dequantize_per_channel.py
```

Local changes:

1. `ax_quantizer.py` imports `ax_quantizer_utils` with a package-relative import.
2. The Conv annotator also matches the `aten.conv2d.padding` overload emitted by
   PyTorch for `Conv2d(padding="same")`. Without this pattern, PP-OCRv6
   `StemBlock.stem2a/stem2b` remains a float Conv island.
3. The HardSigmoid activation annotator is registered and also recognizes
   Paddle's `relu6(1.2 * x + 3) / 6` decomposition. The slope Mul stays
   independently quantized; the activation partition covers `Add -> ReLU6 -> Div`.

Project-specific graph-domain behavior remains in
`pytorchocr.quantization.bridge.AxeraQuantizerAdapter`, outside the copied
quantizer source files.

# QAT.Ultralytics.YOLOv5 Vendor Record (LSQ)

Additional file copied from:

```text
repository: https://github.com/AXERA-TECH/QAT.Ultralytics.YOLOv5
commit:     325fef79
license:    AGPL-3.0
source:     utils/ax_quantizer_lsq.py
```

Vendored source files:

```text
ax_quantizer_lsq.py
```

Local changes:

1. Package-relative import: `from utils.ax_quantizer_utils` ->
   `from .ax_quantizer_utils`.
2. `AXQuantizer.__init__` accepts `config_file` (loaded via `load_config`),
   matching the vendored `ax_quantizer.py` constructor contract.

Supporting logic ported into `ax_quantizer_utils.py` (originally from
QAT.Ultralytics.YOLOv5 `utils/ax_quantizer_utils.py`):

```text
get_weight_shape        resolve weight tensor shape for per-channel channel_len
_ctr_is_fakequat        detect _LearnableFakeQuantize qspec ctrs
get_weight_qspec(quantization_config, weight_node_shape)   LSQ channel_len injection
```

`_LearnableFakeQuantize` itself is used directly from PyTorch 2.6
(`torch.ao.quantization._learnable_fake_quantize`); no local copy.
