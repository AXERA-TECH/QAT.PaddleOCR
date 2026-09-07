import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper


SCRIPT = Path(__file__).resolve().parents[1] / "tools/verify_qat_onnx.py"
SPEC = importlib.util.spec_from_file_location("verify_qat_onnx", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY
SPEC.loader.exec_module(VERIFY)


def make_qdq_linear(weight_dtype: int) -> onnx.ModelProto:
    signed_range = {
        TensorProto.INT8: (-127, 127),
        TensorProto.INT16: (-32767, 32767),
    }
    qmin, qmax = signed_range[weight_dtype]
    weight_values = [qmin, -1, 0, 1, qmax, 2, -2, 3, -3, 4, -4, 5]
    initializers = [
        helper.make_tensor("act_scale", TensorProto.FLOAT, [], [0.25]),
        helper.make_tensor("act_zp", TensorProto.UINT16, [], [0]),
        helper.make_tensor("weight_q", weight_dtype, [4, 3], weight_values),
        helper.make_tensor("weight_scale", TensorProto.FLOAT, [4], [0.1] * 4),
        helper.make_tensor("weight_zp", weight_dtype, [4], [0] * 4),
        helper.make_tensor("bias", TensorProto.FLOAT, [4], [0.0] * 4),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear", ["x", "act_scale", "act_zp"], ["x_q"], name="x_q"
        ),
        helper.make_node(
            "DequantizeLinear",
            ["x_q", "act_scale", "act_zp"],
            ["x_dq"],
            name="x_dq",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zp"],
            ["weight_dq"],
            name="weight_dq",
            axis=0,
        ),
        helper.make_node(
            "Transpose", ["weight_dq"], ["weight_t"], name="weight_t", perm=[1, 0]
        ),
        helper.make_node("MatMul", ["x_dq", "weight_t"], ["matmul"], name="matmul"),
        helper.make_node("Add", ["matmul", "bias"], ["linear"], name="bias_add"),
        helper.make_node(
            "QuantizeLinear",
            ["linear", "act_scale", "act_zp"],
            ["output_q"],
            name="output_q",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "act_scale", "act_zp"],
            ["output"],
            name="output_dq",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    onnx.checker.check_model(model)
    return model


class VerifyQatOnnxTest(unittest.TestCase):
    def assert_linear_passes(self, weight_dtype: int):
        verifier = VERIFY.Verifier(make_qdq_linear(weight_dtype), check_value=True)
        checker = subprocess.CompletedProcess([], 0, "", "")

        findings = verifier.run(checker)

        self.assertEqual(findings, [])
        self.assertEqual([pattern.kind for pattern in verifier.patterns], ["Linear"])

    def test_s16_per_channel_linear_weight(self):
        self.assert_linear_passes(TensorProto.INT16)

    def test_s8_per_channel_linear_weight_remains_supported(self):
        self.assert_linear_passes(TensorProto.INT8)


if __name__ == "__main__":
    unittest.main()
