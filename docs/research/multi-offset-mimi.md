# Multi-Offset Mimi Encoding for Higher Resolution

Research note, 2026-03-27. Deferred.

## Idea

Run Mimi's encoder multiple times with time-shifted input to increase effective temporal resolution. Mimi operates at 12.5 Hz (80ms stride). Running at offset 0ms and 40ms gives interleaved 25 Hz; three offsets (0, 27, 53ms) give 37.5 Hz; four give 50 Hz.

## Why half-stride is optimal for two passes

Half-stride (40ms) maximizes the minimum distance to the nearest original frame, giving the best gap-filling for transition detection. The encoder's receptive field (~150-300ms estimated from kernel sizes) means adjacent interleaved frames are correlated, but the transition signal (codebook 0 flipped or not) still benefits from finer sampling.

## Trade-offs

- Cost scales linearly (each pass is a full encode, ~500x realtime on MLX)
- Diminishing returns from receptive field overlap
- MMS_FA already provides 50 Hz phoneme classification for free
- Most useful as a research tool for understanding Mimi's temporal limits, not a production feature

## When this becomes relevant

If we move to a semantic-codec-based importance scoring approach (SpeechTokenizer at 50 Hz, or X-Codec 2.0) this technique is unnecessary. It's a workaround for Mimi's specific 12.5 Hz limitation. Worth trying if Mimi's semantic signal proves uniquely valuable despite its low resolution.
