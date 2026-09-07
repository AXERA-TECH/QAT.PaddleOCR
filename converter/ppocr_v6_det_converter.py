import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import torch
from converter.weight_mapping import copy_paddle_state_dict_strict
from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPOCRv6DetConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        super(PPOCRv6DetConverter, self).__init__(config, **kwargs)
        self.load_paddle_weights(paddle_pretrained_model_path)
        self.net.eval()

    def load_paddle_weights(self, weights_path):
        print('paddle weights loading...')
        try:
            import paddle.fluid as fluid
            with fluid.dygraph.guard():
                para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        except:
            import paddle
            import sys
            import numpy
            # Workaround for Paddle 3.0 pickle format requiring numpy 2.x APIs
            if 'numpy._core' not in sys.modules:
                import types
                _core = types.ModuleType('numpy._core')
                _core.multiarray = numpy.core.multiarray
                _core._multiarray_umath = numpy.core._multiarray_umath
                _core.numerictypes = numpy.core.numerictypes
                _core.umath = numpy.core.umath
                _core._methods = numpy.core._methods
                _core.fromnumeric = numpy.core.fromnumeric
                sys.modules['numpy._core'] = _core
                sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
                sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
                sys.modules['numpy._core.numerictypes'] = numpy.core.numerictypes
                sys.modules['numpy._core.umath'] = numpy.core.umath
                sys.modules['numpy._core._methods'] = numpy.core._methods
                sys.modules['numpy._core.fromnumeric'] = numpy.core.fromnumeric
            para_state_dict = paddle.load(weights_path)

        report = copy_paddle_state_dict_strict(self.net, para_state_dict)
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
        print('model is loaded: {}'.format(weights_path))


def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))

    return res['Architecture']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    parser.add_argument(
        "--output",
        default="weights/ptocr_v6_small_det_full.pth",
        help="Output path for the complete main+auxiliary training state dict.",
    )
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)
    else:
        raise NotImplementedError

    paddle_pretrained_model_path = os.path.abspath(args.src_model_path)
    converter = PPOCRv6DetConverter(cfg, paddle_pretrained_model_path)

    # Test with random input to verify model forward pass
    np.random.seed(666)
    inputs = np.random.randn(1, 3, 640, 640).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    if isinstance(out, dict):
        out = out['maps']
    out = out.data.numpy()
    print('output shape:', out.shape)
    print('output sum: {:.6f}, mean: {:.6f}, max: {:.6f}, min: {:.6f}'.format(
        np.sum(out), np.mean(out), np.max(out), np.min(out)))

    # save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    converter.save_pytorch_weights(args.output)

    print('done.')
