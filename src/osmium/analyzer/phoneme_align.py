import numpy as np
from osmium.analyzer.importance import ImportanceMap
from osmium.analyzer.phoneme_class import LABEL_TO_CLASS, PHONEME_CLASS_FLOORS


def phoneme_segments_to_importance(
    segments: list[tuple[str, float, float]],
    duration: float,
    frame_rate: float = 50.0,
) -> ImportanceMap:
    n_frames = max(1, int(duration * frame_rate))
    scores = np.full(n_frames, PHONEME_CLASS_FLOORS["silence"], dtype=np.float32)
    times = np.linspace(0, duration, n_frames)

    for label, start, end in segments:
        cls = LABEL_TO_CLASS.get(label, "plosive")
        floor = PHONEME_CLASS_FLOORS[cls]
        i_start = int(start * frame_rate)
        i_end = min(int(end * frame_rate), n_frames)
        scores[i_start:i_end] = floor

    return ImportanceMap(
        scores=scores,
        times=times,
        frame_rate=frame_rate,
        duration=duration,
    )


def analyze_phoneme_aligned(
    samples: np.ndarray,
    sample_rate: int = 24000,
) -> ImportanceMap:
    import torch
    import torchaudio
    from torchaudio.pipelines import MMS_FA as bundle
    from osmium.analyzer.phoneme_class import _load_model

    duration = len(samples) / sample_rate
    target_sr = 16000

    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)

    try:
        import whisper
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(
            samples.astype(np.float32), language="en",
        )
        transcript = result["text"].strip()
    except ImportError:
        raise ImportError(
            "--phoneme-align requires whisper: uv pip install openai-whisper"
        )

    if not transcript:
        return phoneme_segments_to_importance([], duration)

    model = _load_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.no_grad():
        emission, _ = model(waveform)

    tokens = tokenizer(transcript)
    token_spans = aligner(emission[0], torch.tensor(tokens))

    labels = bundle.get_labels()
    frame_dur = duration / emission.shape[1]
    segments = []
    for span in token_spans:
        label = labels[span.token]
        start = span.start * frame_dur
        end = (span.end + 1) * frame_dur
        segments.append((label, start, end))

    return phoneme_segments_to_importance(segments, duration)
