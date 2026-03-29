import argparse
from pathlib import Path

import numpy as np
import torch


def extract_mlx_weights(backbone_sd: dict, head_sd: dict) -> list[tuple[str, np.ndarray]]:
    weights = []

    def _c(name, tensor):
        weights.append((name, tensor.detach().cpu().numpy()))

    _c("embed.weight", backbone_sd["embed.weight"].transpose(1, 2))
    _c("embed.bias", backbone_sd["embed.bias"])
    _c("norm.weight", backbone_sd["norm.weight"])
    _c("norm.bias", backbone_sd["norm.bias"])

    for i in range(8):
        pt = f"convnext.{i}"
        _c(f"{pt}.dwconv.weight", backbone_sd[f"{pt}.dwconv.weight"].transpose(1, 2))
        _c(f"{pt}.dwconv.bias", backbone_sd[f"{pt}.dwconv.bias"])
        _c(f"{pt}.norm.weight", backbone_sd[f"{pt}.norm.weight"])
        _c(f"{pt}.norm.bias", backbone_sd[f"{pt}.norm.bias"])
        _c(f"{pt}.pwconv1.weight", backbone_sd[f"{pt}.pwconv1.weight"])
        _c(f"{pt}.pwconv1.bias", backbone_sd[f"{pt}.pwconv1.bias"])
        _c(f"{pt}.pwconv2.weight", backbone_sd[f"{pt}.pwconv2.weight"])
        _c(f"{pt}.pwconv2.bias", backbone_sd[f"{pt}.pwconv2.bias"])
        _c(f"{pt}.gamma", backbone_sd[f"{pt}.gamma"])

    _c("final_layer_norm.weight", backbone_sd["final_layer_norm.weight"])
    _c("final_layer_norm.bias", backbone_sd["final_layer_norm.bias"])
    _c("head_out.weight", head_sd["out.weight"])
    _c("head_out.bias", head_sd["out.bias"])

    return weights


def convert_checkpoint(checkpoint_path: Path, output_dir: Path):
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    backbone_sd = {}
    head_sd = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
        elif k.startswith("head."):
            head_sd[k[len("head."):]] = v

    weights = extract_mlx_weights(backbone_sd, head_sd)

    weight_dict = {name: arr for name, arr in weights}
    output_path = output_dir / "weights.npz"
    np.savez(str(output_path), **weight_dict)

    print(f"MLX weights saved to {output_path}")
    print(f"  {len(weights)} arrays, {sum(v.nbytes for _, v in weights) / 1e6:.1f} MB")

    from osmium.tsm.vocos_mlx import VocosMLX
    model = VocosMLX()
    mlx_weights = [(name, mx.array(arr)) for name, arr in weights]
    model.load_weights(mlx_weights, strict=False)
    mx.eval(model.parameters())
    print("Verification: MLX model loaded successfully")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training/models/vocos-mel-24khz-finetuned"))
    args = parser.parse_args()
    convert_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
