import pytest
import torch
import numpy as np


def test_convert_weights_produces_correct_keys():
    from scripts.vocos_finetune.convert_mlx import extract_mlx_weights
    from vocos.pretrained import Vocos
    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    weights = extract_mlx_weights(pt_model.backbone.state_dict(), pt_model.head.state_dict())
    keys = [k for k, v in weights]
    assert "embed.weight" in keys
    assert "embed.bias" in keys
    assert "convnext.0.dwconv.weight" in keys
    assert "convnext.7.gamma" in keys
    assert "head_out.weight" in keys
    assert "final_layer_norm.weight" in keys


def test_convert_weights_shapes_match_mlx_model():
    from scripts.vocos_finetune.convert_mlx import extract_mlx_weights
    from vocos.pretrained import Vocos
    pt_model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    weights = extract_mlx_weights(pt_model.backbone.state_dict(), pt_model.head.state_dict())
    weight_dict = dict(weights)
    assert weight_dict["embed.weight"].shape == (512, 7, 100)
    assert weight_dict["head_out.weight"].shape == (1026, 512)
    assert weight_dict["convnext.0.gamma"].shape == (512,)
