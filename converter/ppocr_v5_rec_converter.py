# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20
from converter.weight_mapping import copy_paddle_state_dict_strict

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))

class PPOCRv5RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        super(PPOCRv5RecConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()

    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
        report = copy_paddle_state_dict_strict(
            self.net,
            para_state_dict,
            source_transpose_suffixes=(
                'fc1.weight',
                'fc2.weight',
                'fc.weight',
                'qkv.weight',
                'proj.weight',
                'out_proj.weight',
                'q.weight',
                'kv.weight',
                'tgt_word_prj.weight',
            ),
        )
        print(
            'strict weight mapping: source={}, target={}, copied={}, '
            'allowed_missing={}, ignored_source={}'.format(
                report.source_count,
                report.target_count,
                report.copied_count,
                len(report.allowed_missing_targets),
                report.ignored_source_count,
            )
        )
        print('model is loaded.')

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))
    if res['Architecture']['Head']['name'] == 'MultiHead':
        char_dict_path = os.path.abspath(res['Global']['character_dict_path'])
        if not os.path.exists(char_dict_path):
            raise FileNotFoundError('{} is not existed.'.format(char_dict_path))
        character_str = []
        with open(char_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_str.append(line)
        use_space_char = res['Global']['use_space_char']
        if use_space_char:
            character_str.append(" ")
        character_str = ['blank'] + character_str
        char_num = len(character_str)
        res['Architecture']['Head']['out_channels_list'] = {
            'CTCLabelDecode': char_num,
            'SARLabelDecode': char_num + 2,
            'NRTRLabelDecode': char_num + 3
        }
    return res['Architecture']

if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument(
        "--output",
        default="weights/ptocr_v5_mobile_rec_full.pth",
        help="Output path for the complete CTC+NRTR PyTorch state dict.",
    )
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)

    else:
        raise NotImplementedError

    converter = PPOCRv5RecConverter(cfg, args.src_model_path)

    np.random.seed(666)
    inputs = np.random.randn(1,3,48,320).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    converter.save_pytorch_weights(args.output)
    print('done.')
