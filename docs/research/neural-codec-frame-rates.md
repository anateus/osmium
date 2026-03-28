# Neural Audio Codec Frame Rates and Semantic Codebooks

Research compiled for evaluating alternatives to Mimi for importance scoring.

Last updated: 2026-03-27

---

## Context

Mimi operates at 12.5 Hz (80ms frames). This is too coarse for consonant-level importance scoring — a plosive burst spanning 5-15ms falls within a single frame. We investigated whether higher frame rate codecs with semantic-aware codebooks exist.

## Frame Rate Comparison

| Codec | Frame Rate (Hz) | Sample Rate | Codebooks | Semantic First CB | Bitrate | Open Weights |
|---|---|---|---|---|---|---|
| Mimi (Kyutai) | 12.5 | 24 kHz | 8 RVQ | Yes (WavLM) | 1.1 kbps | Yes |
| EnCodec 24kHz (Meta) | 75 | 24 kHz | 2-32 | No | 1.5-24 kbps | Yes |
| EnCodec 48kHz (Meta) | 150 | 48 kHz | variable | No | 3-24 kbps | Yes |
| SoundStream (Google) | 50 | 24 kHz | 3-32 RVQ | No | 3-18 kbps | No |
| DAC 44kHz (Descript) | ~86 | 44.1 kHz | 9 | No | ~8 kbps | Yes |
| DAC 24kHz (Descript) | ~75 | 24 kHz | 9 | No | ~8 kbps | Yes |
| SNAC 24kHz | 12/23/47 | 24 kHz | 3 (multi-scale) | No | 0.98 kbps | Yes (MIT) |
| SNAC 44kHz | 14/29/57/115 | 44.1 kHz | 4 (multi-scale) | No | 2.6 kbps | Yes (MIT) |
| SpeechTokenizer | 50 | 16 kHz | 8 RVQ | Yes (HuBERT layer 9) | ~4 kbps | Yes |
| X-Codec 2.0 | 50 | 16 kHz | 1 (single) | Yes (Wav2Vec2-BERT/HuBERT) | ~0.6 kbps | Yes |
| X-Codec 2.0 improved | 25 | 24 kHz | 1 (single) | Yes (frozen HuBERT) | ~0.3 kbps | Yes |
| DualCodec | 12.5 or 25 | 24 kHz | 8 RVQ | Yes (w2v-BERT 2.0) | variable | Yes |
| WavTokenizer | 40 or 75 | 24 kHz | 1 (single FSQ) | Implicit | ~0.5-0.9 kbps | Yes |
| BigCodec | 80 | 16 kHz | 1 (8192 codes) | No | 1.04 kbps | Yes |
| OmniCodec | 12.5 or 6.25 | 24 kHz | hierarchical | Yes | variable | Paper only |
| FlexiCodec | 3-12.5 (dynamic) | 24 kHz | multi-codebook | Yes (ASR features) | variable | Yes |

## Codecs with Semantic-Aware First Codebooks

| Codec | Semantic Method | Teacher Model |
|---|---|---|
| Mimi | Distillation loss on codebook 0 | WavLM |
| SpeechTokenizer | Distillation on RVQ-1 | HuBERT (layer 9) |
| X-Codec 2.0 | Semantic encoder + codec fusion | Wav2Vec2-BERT / HuBERT |
| DualCodec | Dual-stream encoding (SSL path) | w2v-BERT 2.0 (layer 16) |
| OmniCodec | Semantic-acoustic disentanglement | Pre-trained audio understanding model |
| FlexiCodec | ASR-feature dual stream | ASR features |

## Best Candidates for Osmium

For importance scoring with both semantic awareness and consonant-level resolution:

1. **SpeechTokenizer** (50 Hz, HuBERT-distilled RVQ-1) — same RVQ-8 structure as Mimi, 4x the temporal resolution. Frame rate matches MMS_FA exactly. Open weights.

2. **X-Codec 2.0** (50 Hz, single semantic codebook) — simpler architecture, open weights. Single codebook carries both semantic and acoustic info.

3. **DualCodec** (25 Hz, w2v-BERT 2.0) — 2x Mimi's resolution, strong semantic properties from dual-stream design.

## Decision (2026-03-27)

Deferred. The current mel importance path (93.8 Hz with peak spreading, HF energy, silence protection) plus MMS_FA phoneme-class floors (50 Hz) already provides good consonant-level importance scoring. A semantic codec would add value mainly for linguistically-informed importance (content words vs function words) — a subtler effect than the consonant/vowel/pause distinction we already capture. Revisit if the mel+phoneme path hits quality ceilings.

## References

- [SpeechTokenizer GitHub](https://github.com/ZhangXInFD/SpeechTokenizer)
- [X-Codec 2.0 GitHub](https://github.com/zhenye234/X-Codec-2.0)
- [DualCodec GitHub](https://github.com/jiaqili3/DualCodec)
- [Kyutai Codec Explainer](https://kyutai.org/codec-explainer)
- [EnCodec GitHub](https://github.com/facebookresearch/encodec)
- [DAC GitHub](https://github.com/descriptinc/descript-audio-codec)
- [SNAC GitHub](https://github.com/hubertsiuzdak/snac)
