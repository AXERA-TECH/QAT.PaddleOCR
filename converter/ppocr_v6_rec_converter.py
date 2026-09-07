import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy

import numpy as np
import torch

from converter.weight_mapping import copy_paddle_state_dict_strict
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPOCRv6RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(
            paddle_pretrained_model_path
        )
        out_channels_list = self.get_out_channels_list(para_state_dict)
        out_channels = out_channels_list["CTCLabelDecode"]
        print("out_channels: ", out_channels)
        config = copy.deepcopy(config)
        if config["Head"]["name"] == "MultiHead":
            config["Head"]["out_channels_list"] = out_channels_list
        kwargs["out_channels"] = out_channels
        super(PPOCRv6RecConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print("model is loaded: {}".format(paddle_pretrained_model_path))
        self.net.eval()

    def get_out_channels_list(self, para_state_dict):
        ctc_out_channels = self.get_ctc_out_channels(para_state_dict)
        return {
            "CTCLabelDecode": ctc_out_channels,
            "SARLabelDecode": self.get_sar_out_channels(
                para_state_dict, ctc_out_channels
            ),
            "NRTRLabelDecode": self.get_nrtr_out_channels(
                para_state_dict, ctc_out_channels
            ),
        }

    def get_ctc_out_channels(self, para_state_dict):
        for name in ("head.ctc_head.fc2.bias", "head.ctc_head.fc.bias"):
            if name in para_state_dict:
                return para_state_dict[name].shape[0]
        for key, value in para_state_dict.items():
            if key.startswith("head.ctc_head.") and key.endswith(".bias"):
                return value.shape[0]
        raise KeyError("Cannot infer CTC out_channels from Paddle weights.")

    def get_sar_out_channels(self, para_state_dict, default_out_channels):
        for key, value in para_state_dict.items():
            if key.startswith("head.sar_head.") and key.endswith(".bias"):
                return value.shape[0]
        return default_out_channels + 2

    def get_nrtr_out_channels(self, para_state_dict, default_out_channels):
        for name in (
            "head.gtc_head.embedding.embedding.weight",
            "head.gtc_head.tgt_word_prj.weight",
        ):
            if name in para_state_dict:
                shape = para_state_dict[name].shape
                # Paddle's NRTR Transformer adds one vocabulary slot to the
                # configured NRTRLabelDecode channel count.
                vocab_size = shape[0] if name.endswith("embedding.weight") else shape[-1]
                return vocab_size - 1
        return default_out_channels + 4

    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        report = copy_paddle_state_dict_strict(
            self.net,
            para_state_dict,
            source_transpose_suffixes=(
                "fc1.weight",
                "fc2.weight",
                "fc.weight",
                "qkv.weight",
                "proj.weight",
                "out_proj.weight",
                "q.weight",
                "kv.weight",
                "tgt_word_prj.weight",
            ),
        )
        print(
            "strict weight mapping: source={}, target={}, copied={}, "
            "allowed_missing={}, ignored_source={}".format(
                report.source_count,
                report.target_count,
                report.copied_count,
                len(report.allowed_missing_targets),
                report.ignored_source_count,
            )
        )
        print("model is loaded.")

    def get_inference_state_dict(self):
        skip_prefixes = ("head.before_gtc.", "head.gtc_head.")
        return {
            k: v
            for k, v in self.net.state_dict().items()
            if not k.startswith(skip_prefixes)
        }

    def save_inference_pytorch_weights(self, weights_path):
        state_dict = self.get_inference_state_dict()
        removed = len(self.net.state_dict()) - len(state_dict)
        try:
            torch.save(state_dict, weights_path, _use_new_zipfile_serialization=False)
        except TypeError:
            torch.save(state_dict, weights_path)
        print("model is saved: {}".format(weights_path))
        print("inference-only state_dict removed {} GTC keys.".format(removed))


def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("{} is not existed.".format(yaml_path))
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        res = yaml.safe_load(f)
    if res.get("Architecture") is None:
        raise ValueError("{} has no Architecture".format(yaml_path))
    if res["Architecture"]["Head"]["name"] == "MultiHead":
        char_dict_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            res["Global"]["character_dict_path"],
        )
        if not os.path.exists(char_dict_path):
            raise FileNotFoundError("{} is not existed.".format(char_dict_path))
        character_str = []
        with open(char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                character_str.append(line)
        use_space_char = res["Global"]["use_space_char"]
        if use_space_char:
            character_str.append(" ")
        character_str = ["blank"] + character_str
        char_num = len(character_str)
        res["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }
    return res["Architecture"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        type=str,
        help="Assign the yaml path of network configuration",
        default=None,
    )
    parser.add_argument(
        "--src_model_path",
        type=str,
        help="Assign the paddleOCR trained model(best_accuracy)",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        choices=["full", "inference"],
        default="full",
        help="Export the full CTC+NRTR training state by default.",
    )
    parser.add_argument(
        "--output",
        default="weights/ptocr_v6_small_rec_full.pth",
        help="Output path for the converted PyTorch state dict.",
    )
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError("{} is not existed.".format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)
    else:
        raise NotImplementedError

    converter = PPOCRv6RecConverter(cfg, args.src_model_path)

    # Test with random input to verify model forward pass
    np.random.seed(666)
    inputs = np.random.randn(1, 3, 48, 320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    print("output shape:", out.shape)
    print(
        "output sum: {:.6f}, mean: {:.6f}, max: {:.6f}, min: {:.6f}".format(
            np.sum(out), np.mean(out), np.max(out), np.min(out)
        )
    )

    # save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    if args.save_mode == "inference":
        converter.save_inference_pytorch_weights(args.output)
    else:
        converter.save_pytorch_weights(args.output)

    print("done.")
