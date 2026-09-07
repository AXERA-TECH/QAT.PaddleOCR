import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = ROOT / ".codex/skills/ppocr-pulsar2-config/scripts"
GENERATOR = SKILL_SCRIPTS / "generate_pulsar2_config.py"
VALIDATOR = SKILL_SCRIPTS / "validate_pulsar2_config.py"


def write_identity_model(path: Path, node_name: str = "identity") -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name=node_name)],
        "generic_pulsar2",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )
    path.write_bytes(model.SerializeToString())


class Pulsar2ConfigSkillTest(unittest.TestCase):
    def run_script(self, script: Path, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_generates_and_validates_generic_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "model.onnx"
            config_path = root / "config.json"
            write_identity_model(onnx_path)

            generated = self.run_script(
                GENERATOR,
                "--onnx",
                str(onnx_path),
                "--output",
                str(config_path),
                "--calibration-dataset",
                str(root / "calibration.zip"),
            )

            self.assertEqual(generated.returncode, 0, generated.stderr)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["input"], str(onnx_path.resolve()))
            self.assertEqual(config["quant"]["layer_configs"], [])
            self.assertEqual(config["input_processors"][0]["tensor_name"], "input")

            validated = self.run_script(
                VALIDATOR,
                "--onnx",
                str(onnx_path),
                "--config",
                str(config_path),
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn('"status": "pass"', validated.stdout)

    def test_profile_entrypoint_is_available_and_model_specific(self):
        help_result = self.run_script(GENERATOR, "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("ppocrv5-rec", help_result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "model.onnx"
            config_path = root / "config.json"
            write_identity_model(onnx_path)
            generated = self.run_script(
                GENERATOR,
                "--profile",
                "ppocrv5-rec",
                "--onnx",
                str(onnx_path),
                "--output",
                str(config_path),
                "--calibration-dataset",
                str(root / "calibration.zip"),
            )
            self.assertNotEqual(generated.returncode, 0)
            self.assertIn("Expected FP32 input shape", generated.stderr)

    def test_validates_explicit_layer_rules_against_frontend_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx_path = root / "model.onnx"
            frontend_path = root / "optimized.onnx"
            layer_path = root / "layers.json"
            config_path = root / "config.json"
            write_identity_model(onnx_path, "source_identity")
            write_identity_model(frontend_path, "frontend_identity")
            layer_path.write_text(
                json.dumps([{"layer_names": ["frontend_identity"], "output_data_type": "S8"}]),
                encoding="utf-8",
            )

            generated = self.run_script(
                GENERATOR,
                "--onnx",
                str(onnx_path),
                "--output",
                str(config_path),
                "--layer-configs",
                str(layer_path),
                "--frontend-onnx",
                str(frontend_path),
                "--calibration-dataset",
                str(root / "calibration.zip"),
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)

            validated = self.run_script(
                VALIDATOR,
                "--onnx",
                str(onnx_path),
                "--config",
                str(config_path),
                "--frontend-onnx",
                str(frontend_path),
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)


if __name__ == "__main__":
    unittest.main()
