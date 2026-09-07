import argparse
import json
import logging
from pathlib import Path

import numpy as np
import paddle
import torch

from compare_float_frameworks import (
    DifferenceAccumulator,
    build_dataloader,
    build_route2,
    configure_eval,
    load_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare route2 eager and deploy-reparameterized float models."
    )
    parser.add_argument("task", choices=("det", "rec"))
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--disable-torch-tf32",
        action="store_true",
        help="Disable CUDA TF32 for the eager/reparameterized comparison.",
    )
    return parser.parse_args()


def tensor_output(model, images, task):
    value = model(images)
    if task == "det" and isinstance(value, dict):
        value = value["maps"]
    return value


def output(model, images, task):
    return tensor_output(model, images, task).detach().cpu().numpy()


def main(args):
    args.task = args.task
    if args.disable_torch_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = configure_eval(load_yaml(args.data_config), args)
    config["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    paddle.set_device(args.device.replace("cuda", "gpu"))
    logger = logging.getLogger("reparameterization-compare")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    loader = build_dataloader(
        config, "Eval", args.device.replace("cuda", "gpu"), logger, seed=0
    )
    device = torch.device(args.device)
    eager = build_route2(
        args.route_config,
        args.torch_weights,
        args.task,
        device,
        reparameterize=False,
    )
    deploy = build_route2(
        args.route_config,
        args.torch_weights,
        args.task,
        device,
        reparameterize=True,
    )
    stats = DifferenceAccumulator()
    recognition_stats = None
    if args.task == "rec":
        from pytorchocr.diagnostics import new_recognition_pair

        recognition_stats = new_recognition_pair()
    samples = 0
    with torch.no_grad():
        for batch in loader:
            images = torch.from_numpy(batch[0].numpy()).to(device)
            eager_tensor = tensor_output(eager, images, args.task)
            deploy_tensor = tensor_output(deploy, images, args.task)
            stats.update(
                eager_tensor.detach().cpu().numpy(),
                deploy_tensor.detach().cpu().numpy(),
                samples,
            )
            if recognition_stats is not None:
                from pytorchocr.diagnostics import update_recognition_pair

                update_recognition_pair(
                    recognition_stats,
                    eager_tensor.detach().cpu(),
                    deploy_tensor.detach().cpu(),
                )
            samples += len(images)
            if args.max_samples is not None and samples >= args.max_samples:
                break
    result = {
        "task": args.task,
        "device": args.device,
        "batch_size": args.batch_size,
        "sample_count": samples,
        "torch_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "eager_vs_reparameterized": stats.result(),
    }
    if recognition_stats is not None:
        from pytorchocr.diagnostics import compute_recognition_pair

        result["recognition_equivalence"] = compute_recognition_pair(
            recognition_stats
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
