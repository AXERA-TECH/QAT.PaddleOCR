import argparse
import json
import logging
from pathlib import Path

import numpy as np
import paddle
import torch

from compare_float_frameworks import (
    build_dataloader,
    build_paddle,
    build_route2,
    configure_eval,
    load_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Localize Paddle/PyTorch recognition float differences by layer."
    )
    parser.add_argument("--paddle-config", required=True)
    parser.add_argument("--route-config", required=True)
    parser.add_argument("--paddle-weights", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--sample-index", type=int, action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--disable-torch-tf32", action="store_true")
    return parser.parse_args()


def stats(reference, candidate):
    difference = np.abs(
        reference.astype(np.float64) - candidate.astype(np.float64)
    )
    return {
        "shape": list(reference.shape),
        "mae": float(difference.mean()),
        "p99": float(np.quantile(difference, 0.99)),
        "max_abs": float(difference.max(initial=0.0)),
    }


def paddle_layers(model, images):
    backbone = model.backbone
    value = backbone.conv1(images)
    values = [value]
    for stage_name in ("blocks2", "blocks3", "blocks4", "blocks5"):
        value = getattr(backbone, stage_name)(value)
        values.append(value)
    for block in backbone.blocks6[:3]:
        value = block(value)
        values.append(value)
    value = backbone.blocks6[3].dw_conv(value)
    values.append(value)
    value = backbone.blocks6[3].pw_conv(value)
    values.append(value)
    value = paddle.nn.functional.avg_pool2d(value, [3, 2])
    values.append(value)
    encoded = model.head.ctc_encoder(value)
    logits = model.head.ctc_head(encoded)
    return [item.numpy() for item in (*values, encoded, logits)]


def torch_layers(wrapper, images):
    model = wrapper.model
    backbone = model.backbone
    value = backbone.conv1(images)
    values = [value]
    for stage_name in ("blocks2", "blocks3", "blocks4", "blocks5"):
        value = getattr(backbone, stage_name)(value)
        values.append(value)
    for block in backbone.blocks6[:3]:
        value = block(value)
        values.append(value)
    value = backbone.blocks6[3].dw_conv(value)
    values.append(value)
    value = backbone.blocks6[3].pw_conv(value)
    values.append(value)
    value = torch.nn.functional.avg_pool2d(value, [3, 2])
    values.append(value)
    encoded = model.head.ctc_encoder(value)
    logits = model.head.ctc_head(encoded)
    return [
        item.detach().cpu().numpy() for item in (*values, encoded, logits)
    ]


def cross_feed_pointwise(paddle_model, torch_wrapper, paddle_dw, torch_dw):
    paddle_pw = paddle_model.backbone.blocks6[3].pw_conv
    torch_pw = torch_wrapper.model.backbone.blocks6[3].pw_conv
    torch_on_paddle = torch_pw(
        torch.from_numpy(paddle_dw.numpy()).to(next(torch_pw.parameters()).device)
    )
    paddle_on_torch = paddle_pw(
        paddle.to_tensor(torch_dw.detach().cpu().numpy())
    )
    return torch_on_paddle.detach().cpu().numpy(), paddle_on_torch.numpy()


def main(args):
    args.task = "rec"
    if args.disable_torch_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    config = configure_eval(load_yaml(args.paddle_config), args)
    config["Eval"]["loader"]["batch_size_per_card"] = args.batch_size
    paddle_device = args.device.replace("cuda", "gpu")
    paddle.set_device(paddle_device)
    logger = logging.getLogger("rec-layer-diagnostic")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    loader = build_dataloader(config, "Eval", paddle_device, logger, seed=0)
    paddle_model = build_paddle(
        config, args.route_config, args.paddle_weights, "rec"
    )
    torch_model = build_route2(
        args.route_config, args.torch_weights, "rec", torch.device(args.device)
    )

    target_indices = sorted(set(args.sample_index))
    reports = {}
    names = (
        "conv1",
        "blocks2",
        "blocks3",
        "blocks4",
        "blocks5",
        "blocks6.0",
        "blocks6.1",
        "blocks6.2",
        "blocks6.3.dw_conv",
        "blocks6.3.pw_conv",
        "backbone_pool",
        "ctc_encoder",
        "ctc_logits",
    )
    with paddle.no_grad(), torch.no_grad():
        for batch_index, batch in enumerate(loader):
            start = batch_index * args.batch_size
            stop = start + len(batch[0])
            selected = [index for index in target_indices if start <= index < stop]
            if not selected:
                continue
            paddle_values = paddle_layers(paddle_model, batch[0])
            torch_values = torch_layers(
                torch_model,
                torch.from_numpy(batch[0].numpy()).to(args.device),
            )
            paddle_dw = paddle.to_tensor(paddle_values[8])
            torch_dw = torch.from_numpy(torch_values[8]).to(args.device)
            torch_on_paddle, paddle_on_torch = cross_feed_pointwise(
                paddle_model, torch_model, paddle_dw, torch_dw
            )
            for sample_index in selected:
                local_index = sample_index - start
                reports[str(sample_index)] = {
                    name: stats(reference[local_index], candidate[local_index])
                    for name, reference, candidate in zip(
                        names, paddle_values, torch_values
                    )
                }
                reports[str(sample_index)]["pointwise_cross_feed"] = {
                    "same_paddle_input": stats(
                        paddle_values[9][local_index],
                        torch_on_paddle[local_index],
                    ),
                    "same_torch_input": stats(
                        paddle_on_torch[local_index],
                        torch_values[9][local_index],
                    ),
                    "native_input_delta": stats(
                        paddle_values[8][local_index],
                        torch_values[8][local_index],
                    ),
                }
            if len(reports) == len(target_indices):
                break
    missing = sorted(set(target_indices) - {int(index) for index in reports})
    if missing:
        raise RuntimeError(f"Samples not found: {missing}")
    result = {
        "device": args.device,
        "batch_size": args.batch_size,
        "torch_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "sample_indices": target_indices,
        "layers": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
