import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".codex/skills/ppocr-qat-config-discovery/scripts/"
    / "discover_ppocr_qat_config.py"
)
SPEC = importlib.util.spec_from_file_location("ppocr_qat_config_discovery", SCRIPT)
DISCOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DISCOVERY)


def base_config():
    return {
        "global_config": {
            "is_symmetric": False,
            "input": {"dtype": "U16", "qmin": 0, "qmax": 65535},
            "weight": {"dtype": "S16", "qmin": -32767, "qmax": 32767},
        },
        "regional_configs": [],
    }


def base_u8s8_config():
    return {
        "global_config": {
            "is_symmetric": False,
            "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
            "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
        },
        "regional_configs": [],
    }


def attention_regions():
    return [
        {
            "qkv_linear": "linear",
            "scale_mul": "mul",
            "first_matmul": "matmul",
            "softmax": "softmax",
            "second_matmul": "matmul_1",
        },
        {
            "qkv_linear": "linear_4",
            "scale_mul": "mul_1",
            "first_matmul": "matmul_2",
            "softmax": "softmax_1",
            "second_matmul": "matmul_3",
        },
    ]


class QatConfigDiscoverySkillTest(unittest.TestCase):
    def test_model_without_attention_keeps_global_only_config(self):
        generated = DISCOVERY.update_config(
            base_config(),
            {"attention": [], "fx_nodes": 12},
            "S16",
        )

        self.assertEqual(generated["regional_configs"], [])

    def test_attention_regions_are_generated_as_s16(self):
        generated = DISCOVERY.update_config(
            base_config(),
            {"attention": attention_regions(), "fx_nodes": 100},
            "S16",
        )

        regional = generated["regional_configs"]
        self.assertEqual(len(regional), 5)
        self.assertEqual(regional[0]["module_names"], ["linear", "linear_4"])
        self.assertEqual(regional[0]["module_config"]["input"]["dtype"], "U16")
        self.assertEqual(regional[0]["module_config"]["output"]["dtype"], "S16")
        self.assertEqual(regional[0]["module_config"]["weight"]["dtype"], "S16")
        self.assertEqual(regional[1]["module_config"]["input"]["qmin"], -32767)
        self.assertEqual(regional[3]["module_config"]["output"]["qmax"], 32767)
        self.assertEqual(regional[4]["module_config"]["input"]["dtype"], "S16")
        self.assertEqual(regional[4]["module_config"]["output"]["dtype"], "U16")

    def test_auto_uses_s16_attention_for_global_u16(self):
        generated = DISCOVERY.update_config(
            base_config(),
            {"attention": attention_regions(), "fx_nodes": 100},
            "auto",
        )

        self.assertEqual(
            [group["module_config"]["output"]["dtype"] for group in generated["regional_configs"]],
            ["S16", "S16", "S16", "S16", "U16"],
        )

    def test_auto_keeps_global_u8s8_compatible(self):
        generated = DISCOVERY.update_config(
            base_u8s8_config(),
            {"attention": attention_regions(), "fx_nodes": 100},
            "auto",
        )

        regional = generated["regional_configs"]
        self.assertEqual(regional[0]["module_config"]["input"]["dtype"], "U8")
        self.assertEqual(regional[0]["module_config"]["weight"]["dtype"], "S8")
        self.assertEqual(
            [group["module_config"]["output"]["dtype"] for group in regional],
            ["S8", "S8", "S8", "S8", "U8"],
        )

    def test_explicit_s16_attention_remains_available_with_global_u8s8(self):
        generated = DISCOVERY.update_config(
            base_u8s8_config(),
            {"attention": attention_regions(), "fx_nodes": 100},
            "S16",
        )

        regional = generated["regional_configs"]
        self.assertEqual(regional[0]["module_config"]["input"]["dtype"], "U8")
        self.assertEqual(regional[0]["module_config"]["output"]["dtype"], "S16")
        self.assertEqual(regional[4]["module_config"]["output"]["dtype"], "U8")

    def test_s8_remains_available_for_explicit_comparison(self):
        self.assertEqual(
            DISCOVERY.signed_qspec("S8"),
            {"dtype": "S8", "qmin": -127, "qmax": 127},
        )


if __name__ == "__main__":
    unittest.main()
