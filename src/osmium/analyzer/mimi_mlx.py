import numpy as np
from osmium.analyzer.mimi import MimiCodes, MIMI_FRAME_RATE

HF_REPO = "kyutai/moshiko-mlx-q4"
HF_TOKENIZER = "tokenizer-e351c8d8-checkpoint125.safetensors"

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model

    import mlx.core as mx
    from huggingface_hub import hf_hub_download
    from moshi_mlx.models.mimi import Mimi, mimi_202407

    cfg = mimi_202407(8)
    model = Mimi(cfg)

    weights_path = hf_hub_download(HF_REPO, HF_TOKENIZER)
    _load_weights_filtered(model, weights_path)

    mx.eval(model.parameters())
    model.warmup()

    _model = model
    return model


def _load_weights_filtered(model, weights_path):
    import mlx.core as mx

    raw_weights = mx.load(weights_path)
    max_rvq_rest = 6

    filtered = {}
    for k, v in raw_weights.items():
        skip = False
        if "rvq_rest.vq.layers." in k:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        if idx > max_rvq_rest:
                            skip = True
                    except ValueError:
                        pass
        if ("cluster_usage" in k or "embedding_sum" in k or "_initialized" in k) and skip:
            pass  # only skip training state for layers we're already skipping
        if not skip:
            filtered[k] = v

    model.load_pytorch_weights.__func__  # just verify it exists
    # Manually replicate load_pytorch_weights logic but with our filtered dict
    from moshi_mlx.modules import EuclideanCodebook, ConvTranspose1d

    weights = []
    for k, v in filtered.items():
        k = ".".join([s.removeprefix("_") for s in k.split(".")])
        if k.startswith("encoder.model."):
            k = k.replace("encoder.model.", "encoder.")
        if k.startswith("decoder.model."):
            k = k.replace("decoder.model.", "decoder.")
        if k.endswith(".in_proj_weight"):
            k = k.replace(".in_proj_weight", ".in_proj.weight")
        if k.endswith(".linear1.weight"):
            k = k.replace(".linear1.weight", ".gating.linear1.weight")
        if k.endswith(".linear2.weight"):
            k = k.replace(".linear2.weight", ".gating.linear2.weight")

        for layerIdx, decoderIdx in enumerate([2, 5, 8, 11]):
            k = k.replace(f"decoder.{decoderIdx}.", f"decoder.layers.{layerIdx}.upsample.")
            k = k.replace(f"decoder.{decoderIdx + 1}.", f"decoder.layers.{layerIdx}.residuals.0.")
        for layerIdx, encoderIdx in enumerate([1, 4, 7, 10]):
            k = k.replace(f"encoder.{encoderIdx}.", f"encoder.layers.{layerIdx}.residuals.0.")
            k = k.replace(f"encoder.{encoderIdx + 2}.", f"encoder.layers.{layerIdx}.downsample.")

        k = k.replace("decoder.0.", "decoder.init_conv1d.")
        k = k.replace("decoder.14.", "decoder.final_conv1d.")
        k = k.replace("encoder.0.", "encoder.init_conv1d.")
        k = k.replace("encoder.14.", "encoder.final_conv1d.")
        k = k.replace(".block.1.", ".block.0.")
        k = k.replace(".block.3.", ".block.1.")

        if k.endswith(".conv.weight") or k.endswith(".output_proj.weight") or k.endswith(".input_proj.weight"):
            v = v.swapaxes(-1, -2)
        if k.endswith(".convtr.weight"):
            v = v.transpose(1, 2, 0)
        weights.append((k, v))

    model.load_weights(weights, strict=False)

    def _filter_fn(module, name, _):
        if isinstance(module, EuclideanCodebook) and name == "initialized":
            module.update_in_place()
        if isinstance(module, ConvTranspose1d) and name == "weight":
            module.update_in_place()
        return True

    model.filter_and_map(_filter_fn)


def encode_mlx(samples: np.ndarray, sample_rate: int = 24000) -> MimiCodes:
    import mlx.core as mx

    model = _load_model()

    pcm = mx.array(samples.reshape(1, 1, -1))

    codes = model.encode(pcm)
    mx.eval(codes)

    codes_np = np.array(codes)
    if codes_np.ndim == 3:
        codes_np = codes_np.squeeze(0).T

    duration = len(samples) / sample_rate
    return MimiCodes(
        codes=codes_np,
        frame_rate=MIMI_FRAME_RATE,
        sample_rate=sample_rate,
        duration=duration,
    )
