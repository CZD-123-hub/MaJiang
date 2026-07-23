import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supervised.config import PROJECT_ROOT, config


SUBMISSION_LIMIT_MB = 256.0


def resolve_checkpoint_path(checkpoint_spec, control_cfg):
    if checkpoint_spec in ("best", "latest"):
        ckpt_name = "gbmj_policy_best.pth" if checkpoint_spec == "best" else "gbmj_policy_latest.pth"
        return PROJECT_ROOT / control_cfg["checkpoint_dir"] / ckpt_name
    checkpoint_path = Path(checkpoint_spec)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return checkpoint_path


def build_inference_payload(state, fp16=False):
    converted = {}
    for key, value in state["model"].items():
        if fp16 and torch.is_floating_point(value):
            converted[key] = value.detach().to(dtype=torch.float16, device="cpu")
        else:
            converted[key] = value.detach().to(device="cpu")
    return {
        "model": converted,
        "steps": int(state.get("steps", -1)),
        "epoch": int(state.get("epoch", -1)),
        "best_metrics": state.get("best_metrics"),
        "config": state.get("config", config),
        "inference_only": True,
        "dtype": "float16" if fp16 else "float32",
    }


def file_size_mb(path):
    return path.stat().st_size / 1024.0 / 1024.0


def print_size_report(path):
    size_mb = file_size_mb(path)
    status = "OK" if size_mb <= SUBMISSION_LIMIT_MB else "TOO LARGE"
    print(f"saved: {path} ({size_mb:.2f} MB, {status} for {SUBMISSION_LIMIT_MB:.0f} MB limit)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best", help="best/latest or a checkpoint path")
    parser.add_argument("--output", default=None, help="output path for float32 inference checkpoint")
    parser.add_argument("--fp16-output", default=None, help="output path for float16 inference checkpoint")
    args = parser.parse_args()

    control_cfg = config["control"]
    ckpt_path = resolve_checkpoint_path(args.checkpoint, control_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if args.output is None:
        out_path = ckpt_path.with_name(ckpt_path.stem + "_infer.pth")
    else:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path

    if args.fp16_output is None:
        fp16_out_path = ckpt_path.with_name(ckpt_path.stem + "_infer_fp16.pth")
    else:
        fp16_out_path = Path(args.fp16_output)
        if not fp16_out_path.is_absolute():
            fp16_out_path = PROJECT_ROOT / fp16_out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp16_out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(build_inference_payload(state, fp16=False), out_path)
    torch.save(build_inference_payload(state, fp16=True), fp16_out_path)

    print_size_report(out_path)
    print_size_report(fp16_out_path)


if __name__ == "__main__":
    main()
