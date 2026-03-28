# Natural Fast Speech Production: Literature Review and Corpus Survey

Research compiled for the Osmium variable-rate speech acceleration project.

Last updated: 2026-03-27

---

## 1. What Natural Fast Speakers Actually Do

### 1.1 Differential Segment Compression

The central finding across decades of research is that natural fast speech involves **non-uniform compression** of speech segments. Not all sounds shorten equally when speakers talk faster.

**Vowels compress more than consonants.** This is the most robust and consistent finding in the literature. Vowels, being produced with relatively open vocal tracts and fewer biomechanical constraints, have greater durational flexibility. Consonants, particularly stops and fricatives, have articulatory requirements (closure formation, frication noise buildup) that impose minimum duration floors.

- Crystal & House (1988) established through systematic analysis of six speakers reading connected speech that segmental durations are strongly influenced by speaking rate, but the effect is not uniform across segment types. Their series of papers on segmental durations in connected speech signals remains foundational.
- Gay (1978) showed that at fast speaking rates, vowel durations decreased substantially but acoustic vowel targets (formant frequencies at vowel midpoints) were largely preserved. CV transition rates remained essentially unchanged; instead, movement toward the vowel target simply began earlier. This suggests speakers prioritize spectral distinctiveness even when compressing temporally.
- Janse (2004) confirmed that in naturally produced fast speech, vowels shorten relatively more than consonants, and unstressed syllables get shortened to a greater extent or are more likely to be deleted than stressed syllables.

**Stressed syllables resist compression more than unstressed syllables.** When speakers accelerate, unstressed syllables bear the brunt of the compression. Unstressed vowels are reduced or deleted entirely, while stressed vowels maintain more of their duration and spectral identity. This creates a more "spiky" rhythmic profile at fast rates -- stressed syllables become relatively more prominent.

**Pauses are the first casualty.** Before segment durations even start to compress meaningfully, inter-utterance and inter-phrase pauses shrink or disappear. Faster talkers reduce pause duration between sentences dramatically. Within-utterance pauses (hesitation, breathing) also decrease in frequency and duration.

### 1.2 Coarticulation and Reduction Processes

As speaking rate increases, several connected-speech processes become more frequent and more extreme:

- **Increased coarticulation**: Articulatory gestures overlap more in time. The tongue may begin moving toward the next sound before completing the current one. Byrd's work using TIMIT showed that faster speech involves both individual consonant shortening and a relatively linear increase in overlap of articulations.
- **Assimilation**: Sounds become more similar to their neighbors (e.g., /n/ before /p/ may become [m]). This occurs in normal speech but is significantly more frequent in fast speech.
- **Elision/deletion**: Sounds are dropped entirely. Elision is an extreme result of coarticulation whereby two sounds are articulated so closely in time that a sound between them is completely obscured. Unstressed vowels and word-final consonants are most vulnerable.
- **Flapping**: In American English, intervocalic /t/ and /d/ become flaps more consistently at fast rates.
- **Vowel reduction**: Full vowels reduce toward schwa or are deleted entirely in unstressed positions.

Fosler-Lussier & Morgan (1999) showed through analysis of the Switchboard corpus that pronunciations are strongly dependent on speaking rate and word predictability, with changes observable at the phone, syllable, and word level. Greater fluency is associated with more segment shortening and more articulatory overlap.

### 1.3 Prosodic and Rhythmic Changes

Fast speech is not just "normal speech played faster." The prosodic structure changes qualitatively:

- **Fewer and shorter pauses**: Both inter- and intra-utterance pauses decrease.
- **Fewer F0 resets**: Pitch contours become smoother with fewer boundary-marking pitch resets.
- **Simpler pitch accents**: Fewer bitonal pitch accents and more monotonal ones in fast vs. slow speech. The intonation contour is simplified.
- **Increased rhythmic prominence of stressed syllables**: Because unstressed syllables compress more, the stressed-unstressed duration contrast actually increases, making the rhythm more "choppy" or stress-timed.

### 1.4 Professional Fast Speakers

**Auctioneers** deliver 250-400 words per minute (vs. 120-150 wpm conversational). Their technique relies on:
- Highly structured rhythmic patterns ("cadences") where filler words are slurred while key information (numbers) remains clear.
- Syllable reduction: "seventy" becomes "sevny" to reduce syllable count.
- The pattern is essentially the same thing repeated with only numbers changing, reducing the listener's cognitive load.
- Roughly 3 bid numbers articulated per second.

**Horse racing commentators** employ a progressive acceleration strategy: starting slow and relaxed during the early stages, then accelerating to "fever pitch" as the race reaches its climax. This relies on rapid visual processing, structured linguistic templates, and vocal stamina.

**Sports commentators and news readers** typically speak at 150-190 wpm in professional delivery. Baumann & Trouvain (2001) found that less important information (side comments, parentheticals) is marked by faster articulation, while emphasized discourse segments are marked by slower articulation -- a kind of natural non-uniform rate control at the discourse level.

### 1.5 Speech Gaits: Qualitative Rate Changes

Recent work from the Max Planck Institute for Psycholinguistics introduced the EPONA computational model, which revealed that speakers do not smoothly accelerate. Instead, they switch between distinct speech "gaits" analogous to walking vs. running. There are "sudden shifts, indicating boundaries between at least two gaits" -- a qualitative rather than quantitative change in speech motor planning. This means there may be discrete rate regimes rather than a smooth continuum.

### 1.6 Cross-Linguistic Speaking Rate and Information Rate

Pellegrino, Coupe & Marsico (2011) demonstrated across seven languages that there is a negative correlation between information density per syllable and syllable rate. Languages with simpler syllables (e.g., Japanese) are spoken faster; languages with more complex syllables (e.g., Mandarin) are spoken slower. The result is a roughly constant information rate of ~39 bits/second across languages. This has implications for cross-linguistic application of rate-modification algorithms.

---

## 2. Specific Durational Patterns and Compression Ratios

### 2.1 Typical Speaking Rate Ranges

| Context | Rate (syl/sec) | Rate (wpm) |
|---|---|---|
| Slow/careful speech | ~3.0-3.5 | ~100-120 |
| Normal conversational | ~4.0-5.0 | ~120-150 |
| Fast conversational | ~5.5-6.5 | ~160-200 |
| Professional fast (news) | ~6.0-7.0 | ~170-210 |
| Auctioneer/extreme | ~7.0-9.0 | ~250-400 |
| Auditory processing limit | ~9 syl/sec | -- |

### 2.2 Differential Compression by Phoneme Class

Published research consistently shows a hierarchy of compressibility. The following synthesizes findings across multiple studies:

**Most compressible (compress first and most):**
1. Pauses (inter-utterance, inter-phrase)
2. Vowels in unstressed syllables (especially schwa)
3. Long vowels / tense vowels
4. Vowels in stressed syllables

**Least compressible (have articulatory duration floors):**
5. Nasals
6. Liquids/glides
7. Fricatives
8. Plosive closures (especially voiceless)
9. Plosive bursts / releases

### 2.3 Specific Numerical Data

**Vowel/Consonant ratio changes with rate** (from the Korean/English V/C ratio study):
- Voiced coda context, V/C ratio: Fast 0.67, Habitual 0.69, Slow 0.72
- Voiceless coda context, V/C ratio: Fast 0.71, Habitual 0.57, Slow 0.53
- Paradigmatic V/V ratio (vowels before voiced/voiceless): Fast/Habitual ~1.3, Slow ~1.5

**Incompressibility phenomenon**: As rate increases, the combined effects of shortening become smaller and duration asymptotically approaches a minimum value. This is modeled explicitly in the Klatt (1979) duration model:

```
D = D_min + (D_inh - D_min) * f1 * f2 * ... * fn
```

Where D_min is the minimum (incompressible) duration, D_inh is the inherent duration, and f1...fn are multiplicative factors for contextual effects including speaking rate. The key insight: **the compression factor operates on the compressible portion (D_inh - D_min), not on the full duration.** This naturally produces the nonlinear, asymptotic behavior observed in real speech.

**Stop consonants** show less rate-dependent variation than vowels. Plosive closure durations range from 0-150ms, with word-initial positions showing the longest durations. At fast rates, stop releases may be eliminated entirely (unreleased stops), but the closure phase has a floor set by the time needed to build and release intraoral pressure.

**Coda vs. onset consonants**: Coda consonant durations are more strongly correlated with speaking rate than onset consonant durations. Primary stress vowel durations are more strongly correlated with rate than unstressed vowel durations.

**Hungarian fast speech data** (Deme et al., 2024): In fast speech, vowels reduced more in their duration than consonants; long vowels reduced more than short vowels; and duration differences of long and short vowels were compressed (the phonemic length contrast partially neutralized).

### 2.4 Approximate Compression Factors by Segment Class

Based on synthesis of the literature (Crystal & House 1988, Janse 2004, Gay 1978, Klatt 1979, and the V/C ratio studies), approximate compression factors when going from normal to fast speaking rate (~1.5x overall acceleration):

| Segment class | Approx. compression factor | Notes |
|---|---|---|
| Inter-utterance pauses | 0.2-0.4 (80-60% reduction) | Pauses may be eliminated entirely |
| Unstressed vowels (schwa) | 0.4-0.6 (60-40% reduction) | May delete entirely |
| Stressed vowels | 0.5-0.7 (50-30% reduction) | Spectral targets largely preserved |
| Liquids/glides | 0.6-0.8 (40-20% reduction) | Moderate compression |
| Nasals | 0.7-0.85 (30-15% reduction) | Less compressible |
| Fricatives | 0.75-0.9 (25-10% reduction) | Frication noise needs minimum time |
| Plosive closures | 0.8-0.95 (20-5% reduction) | Articulatory floor constrains |
| Plosive bursts/VOT | 0.85-1.0 (15-0% reduction) | Near-incompressible |

These are approximate ranges synthesized from the literature, not from a single controlled experiment. Actual values depend on language, speaker, phonetic context, prosodic position, and overall rate.

---

## 3. Available Corpora and Datasets

### 3.1 Corpora with Explicit Speaking Rate Variation

#### BonnTempo Corpus (BTC)
- **Citation**: Dellwo, V., Steiner, I., Aschenberner, B., Dankovicova, J., & Wagner, P. (2004). BonnTempo-Corpus & BonnTempo-Tools. Proc. Interspeech 2004, Jeju Island, Korea.
- **URL**: https://www.academia.edu/1311501/The_BonnTempo_Corpus_and_Tools
- **What it contains**: Read speech at multiple elicited speaking rates across five languages. Speakers read the same text at different tempos (very slow, slow, normal, fast, very fast). Contains 24,070 syllables and 43,227 C-and V-intervals.
- **Size**: 5 languages, multiple speakers per language, 5 rate conditions per speaker.
- **Why relevant**: This is one of the few corpora explicitly designed to study speaking rate effects on speech rhythm and timing. Same-speaker matched recordings at different rates are ideal for extracting per-phoneme compression ratios. The accompanying BonnTempo-Tools (Praat-based) facilitate analysis of rhythm measures (%V, deltaC, nPVI, rPVI) as a function of rate.

#### IFA Spoken Language Corpus
- **Citation**: van Son, R.J.J.H. et al. (2001). The IFA corpus: a phonemically segmented Dutch "open source" speech database.
- **URL**: https://www.fon.hum.uva.nl/IFA-SpokenLanguageCorpora/IFAcorpus/
- **What it contains**: Hand-segmented Dutch speech from 8 speakers in a variety of speaking styles (read, retold, informal). For each speaker, a fixed text recorded in several "styles" plus spontaneous speech. All material segmented and labeled at the phoneme level, stored in a relational SQL database.
- **Size**: ~50,000 words per speaker (41 min/speaker), 8 speakers total. 1,000 hours of labeling effort.
- **Why relevant**: Open-source (GPL), phoneme-level alignment, multiple speaking styles from same speakers allows rate comparison. Dutch is well-studied for speech timing. SQL access enables systematic queries across segment types and speaking conditions.

### 3.2 Corpora with Natural Rate Variation (Spontaneous Speech)

#### HCRC Map Task Corpus
- **Citation**: Anderson, A. et al. (1991). The HCRC Map Task Corpus. Language and Speech, 34(4).
- **URL**: https://groups.inf.ed.ac.uk/maptask/
- **LDC**: https://catalog.ldc.upenn.edu/LDC93S12
- **What it contains**: 128 unscripted, task-oriented dialogues. Speakers collaborate verbally to reproduce a route from one participant's map to another's. 64 speakers (32F/32M), each in 4 conversations.
- **Size**: ~18 hours of spontaneous speech.
- **Why relevant**: Task-oriented dialogues produce natural speaking rate variation driven by communicative pressure. Controlled for familiarity (friends vs. strangers) and visual contact. The time pressure of the task elicits naturally faster speech in some conditions. Widely used in prosody and timing research.

#### Buckeye Corpus
- **Citation**: Pitt, M.A. et al. The Buckeye Corpus of Conversational Speech.
- **URL**: https://buckeyecorpus.osu.edu/
- **What it contains**: 307,000 words of conversational speech from 40 speakers in Columbus, Ohio. Orthographically transcribed and phonetically labeled. Recorded as sociolinguistic interviews (essentially monologues).
- **Size**: 40 speakers, ~307,000 words total.
- **Why relevant**: Phonetically labeled spontaneous speech with natural rate variation. The phonetic labels (84.3% agreement for stops, 77.8% for vowels) enable segment-level duration analysis. Widely used for studying reduction, deletion, and speaking rate effects in American English.

#### Switchboard Corpus
- **Citation**: Godfrey, J. et al. (1992). SWITCHBOARD: Telephone Speech Corpus for Research and Development.
- **URL**: https://catalog.ldc.upenn.edu/LDC97S62
- **What it contains**: ~2,400 two-sided telephone conversations among 543 speakers (302M/241F) from all US dialect regions. ~260 hours of conversational English.
- **Size**: ~260 hours, 543 speakers.
- **Why relevant**: Large-scale spontaneous conversational corpus with natural rate variation. Fosler-Lussier & Morgan (1999) demonstrated strong effects of speaking rate on pronunciations in this corpus. Phonetic transcriptions available for portions. The casual conversational style produces extreme rate variation and reduction phenomena directly relevant to modeling natural fast speech.

### 3.3 Professionally Read Speech Corpora

#### Boston University Radio News Corpus (BURNC)
- **Citation**: Ostendorf, M. et al. Boston University Radio Speech Corpus.
- **URL**: https://catalog.ldc.upenn.edu/LDC96S36
- **What it contains**: Over 7 hours of radio news stories from seven FM radio news announcers (4M/3F) at WBUR. Includes both broadcast recordings and laboratory recordings where announcers read stories in "radio style" and "non-radio style." Annotated with orthographic transcription, phonetic alignments, POS tags, and prosodic markers (ToBI).
- **Size**: 7+ hours, 7 speakers.
- **Why relevant**: Professional speakers with trained rate control. The dual-style recordings (radio vs. casual) from the same speakers provide natural speaking rate contrasts. Prosodic annotations (ToBI) enable analysis of how pitch accents and phrasing change with rate. The professional delivery style represents a controlled form of fast speech.

#### LibriSpeech / LibriTTS
- **Citation**: Panayotov, V. et al. (2015). Librispeech: An ASR Corpus Based on Public Domain Audio Books. ICASSP 2015.
- **URL**: https://www.openslr.org/12 (LibriSpeech), https://www.openslr.org/60 (LibriTTS)
- **What it contains**: LibriSpeech: ~1000 hours of 16kHz read English from ~2,484 speakers, derived from LibriVox audiobooks. LibriTTS: 585 hours at 24kHz from 2,456 speakers with sentence-level alignment.
- **Size**: 982-1000 hours.
- **Why relevant**: Massive scale, freely available, and audiobook reading naturally produces rate variation between narrative and dialogue sections, between readers, and within utterances. While not designed for rate research, the scale enables data-driven analysis. LibriTTS at 24kHz is better suited for acoustic analysis than 16kHz LibriSpeech.

### 3.4 Phonetically Aligned Reference Corpora

#### TIMIT
- **Citation**: Garofolo, J.S. et al. (1993). DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus.
- **URL**: https://catalog.ldc.upenn.edu/LDC93S1
- **What it contains**: 630 speakers of 8 major American English dialects, each reading 10 phonetically rich sentences. Time-aligned orthographic, phonetic, and word transcriptions at 16kHz/16-bit.
- **Size**: 6,300 utterances, ~5 hours.
- **Why relevant**: The gold standard for phonetic alignment studies. Byrd (1994) used TIMIT to establish baseline segment durations and speaking rate variation across dialect regions and genders. The phonetic transcription system (61 phones, commonly collapsed to 39) provides a taxonomy for segment-class analysis. Although read speech with limited rate variation, the precise alignments make it valuable as a duration reference.

#### VCTK Corpus
- **Citation**: Veaux, C. et al. CSTR VCTK Corpus.
- **URL**: https://datashare.ed.ac.uk/handle/10283/3443
- **What it contains**: 44 hours of recordings from 109 native English speakers, each reading ~400 sentences. 48kHz studio recordings.
- **Size**: 44 hours, 109 speakers.
- **Why relevant**: High sample rate (48kHz) and multi-speaker design. The speaker diversity enables analysis of between-speaker rate variation. Frequently used in speech synthesis and voice conversion research.

### 3.5 Multilingual Corpora

#### GlobalPhone
- **Citation**: Schultz, T. (2002). GlobalPhone: A Multilingual Speech and Text Database. ICASSP.
- **URL**: https://www.csl.uni-bremen.de/GlobalPhone/
- **What it contains**: Read speech in 20 languages with transcriptions and pronunciation dictionaries. IPA-based phone sets consistent across languages. 16kHz/16-bit.
- **Size**: 400+ hours, 2000+ speakers across 20 languages.
- **Why relevant**: Enables cross-linguistic comparison of speaking rate effects. The consistent recording conditions and phone-set conventions allow direct comparison of segment durations across languages, relevant to the Pellegrino et al. (2011) findings on information rate.

---

## 4. Computational Models of Speaking Rate

### 4.1 Klatt (1979) Duration Model

The foundational rule-based model. Each phoneme has:
- **D_inh**: inherent (baseline) duration
- **D_min**: minimum (incompressible) duration
- **f1...fn**: multiplicative factors for contextual effects

```
D = D_min + (D_inh - D_min) * f1 * f2 * ... * fn
```

Factors include: phoneme identity, voicing, position in word, position in sentence, word length, stress, speaking rate, neighboring phonemes.

**Speaking rate is modeled as one multiplicative factor** that compresses the compressible portion (D_inh - D_min). This naturally produces the asymptotic/incompressibility behavior: segments that are already near their minimum duration cannot be compressed further.

**Strengths**: Transparent, interpretable, captures incompressibility.
**Weaknesses**: The multiplicative model is too simple for non-linear interaction effects. Manually tuned parameters.

### 4.2 van Santen (1994) Sum-of-Products (SOP) Model

A more sophisticated extension of Klatt's approach. Duration is modeled as:

```
D = A(phoneme) + B(factor1) * C(factor2) + ...
```

The SOP model uses a category structure (tree) that divides the phonetic space into similar-behaving groups. For each category, a separate sum-of-products equation is fitted. This captures interaction patterns where one factor amplifies (but does not reverse) the effects of other factors.

Factors include: phonemic identity, phonetic context, phrase boundaries, lexical stress, speaking rate.

**Deployed in**: AT&T Bell Laboratories TTS system.
**Strengths**: Better captures factor interactions than Klatt. Data-driven parameter estimation.
**Weaknesses**: Requires careful category-structure design. Still essentially a linear model within each category.

### 4.3 Pfitzinger (1999) Perceived Speech Rate Model

Models perceived local speech rate as a linear combination of local syllable rate and local phone rate:

```
perceived_rate = a * syllable_rate + b * phone_rate
```

This achieves r = 0.91 correlation with perceptual judgments. Notably, F0 measurements did not improve the model. This has implications for time-scale modification: matching the perceived rate of natural fast speech may require targeting this combined metric rather than simple duration ratios.

### 4.4 EPONA: Speech Gaits Model

The Max Planck Institute's computational model (recent) that revealed discrete speech "gaits" rather than continuous rate adjustment. Speakers switch between qualitatively different motor planning modes (analogous to walking vs. running), with sudden transitions between gaits. This implies that effective speech acceleration might need to respect gait boundaries rather than applying continuous compression.

### 4.5 Neural Duration Models

Modern TTS systems (Tacotron, FastSpeech, VITS) learn implicit duration models from data. FastSpeech 2 uses an explicit duration predictor conditioned on phoneme identity, pitch, and energy. These models implicitly capture many of the patterns described above (differential compression, incompressibility) through learned representations, but are not directly interpretable.

### 4.6 Kasparaitis & Beniuse: Automatic Klatt Parameter Estimation

Proposed an iterative algorithm for automatic estimation of Klatt model factors from annotated audio, removing the need for manual parameter tuning. This makes it feasible to extract language-specific and speaker-specific compression parameters from aligned corpora.

---

## 5. Perceptual Studies on Accelerated Speech Intelligibility

### 5.1 Janse (2004): Natural Fast vs. Artificially Compressed

Janse compared three conditions at ~1.56x normal rate:
1. **Natural fast speech** (speakers talking fast)
2. **Linear time-compression** (uniform compression to match the natural fast rate)
3. **Nonlinear selective compression** (following the syllable-level temporal pattern of natural fast speech)

Key finding: **Linearly time-compressed speech was actually easier to process than both natural fast speech and nonlinearly compressed speech.** Word processing speed was slowed down in the selective compression condition relative to linear compression. This is counterintuitive but important: it suggests that the temporal distortions of natural fast speech (vowel reduction, deletion, coarticulation) actually make speech harder to understand, not easier. The advantage of natural fast speech is presumably in the preserved spectral cues, not in the temporal pattern per se.

However, Janse also noted that the only nonlinear aspect that could improve intelligibility at high rates was **pause removal** -- but only when rates were relatively high.

### 5.2 Adank & Janse: Perceptual Learning of Fast Speech

- Significant rapid perceptual learning of time-compressed speech occurs within the first ~20 sentences.
- Transfer of learning works asymmetrically: adapting to time-compressed speech helps with natural fast speech, but not vice versa.
- Older adults adapt to time-compressed speech at comparable magnitude to young adults but fail to transfer learning to different speech rates and show no additional benefit beyond 20 sentences of exposure.
- Learning of time-compressed speech positively correlates with natural fast speech perception.

### 5.3 Auditory Processing Limits

- There appears to be an upper bound to auditory processing of approximately **9 syllables per second** in English.
- Beyond ~2.5-3x compression, intelligibility drops precipitously regardless of method.
- Dupoux & Green (1997) and subsequent work showed that listeners can adapt to surprisingly fast rates with brief exposure, suggesting the limit is partly attentional/cognitive rather than purely auditory.

### 5.4 What Helps Intelligibility at Fast Rates

Based on synthesis of the literature:

1. **Preserving consonant transitions**: The formant transitions between consonants and vowels carry the majority of consonant identity information. The patent by US7065485B1 demonstrates this principle: expanding CV transitions (factor 0.5 = 2x expansion) while compressing steady-state vowels (factor 1.8 = 1.8x compression) and lengthening fricatives (factor 0.8 = 1.25x expansion) maintains overall duration while boosting consonant salience.

2. **Preserving temporal envelope modulations**: Rosen (1992) established that speech intelligibility depends on temporal modulations in three frequency bands: envelope (2-50 Hz, carries manner/voicing), periodicity (50-500 Hz, carries voicing/manner/intonation), and fine structure (600-10kHz, carries place/vowel quality). The 2-50 Hz envelope, which corresponds to syllabic rhythm, is the most critical for intelligibility.

3. **Maintaining stressed syllable prominence**: Because natural fast speech increases the stressed/unstressed duration contrast, preserving this contrast in artificial acceleration should aid intelligibility.

4. **Phoneme-class-adaptive compression** (Haque et al., 2023): The ATSM algorithm performs forced alignment using Montreal Forced Aligner and applies phoneme-cluster-specific compression rates rather than uniform compression. This was evaluated for streaming services (OTT, audiobooks, online lectures) and showed improvements over conventional TSM in diagnostic rhyme tests.

5. **Perceptual learning/adaptation**: Listeners adapt rapidly. Even 20 sentences of exposure to accelerated speech significantly improves comprehension. This suggests that slight imperfections in time-scale modification are tolerable if they are consistent.

### 5.5 What Hurts Intelligibility

1. **Uniform compression beyond 2x**: Quality degrades rapidly.
2. **Compressing consonant bursts and transitions**: These carry the most information-dense cues.
3. **Eliminating the stressed/unstressed contrast**: Making everything the same duration is worse than preserving natural rhythm.
4. **Phase discontinuities and artifacts**: Traditional WSOLA/PSOLA algorithms introduce artifacts under extreme or rapidly varying compression factors. STSM-FiLM (2025) addresses this with a fully neural architecture using FiLM conditioning for continuous speed-factor control.

### 5.6 Natural Fast Speech is Perceived as Faster

Koreman et al. (2016) showed that natural fast speech is perceived as faster than linearly time-compressed speech matched for the same overall rate. This means that the non-uniform compression pattern of natural speech creates a subjective impression of greater speed. For a speed-listening product, this could be either a feature (perceived efficiency) or a bug (perceived difficulty).

---

## 6. Implications for Osmium

### 6.1 Core Design Principles Supported by Literature

1. **Compress pauses first and most aggressively.** This is where the greatest time savings come with the least perceptual cost.

2. **Compress vowels more than consonants.** This matches natural fast speech production and is supported by the incompressibility principle. Unstressed vowels should compress more than stressed vowels.

3. **Protect consonant onsets, bursts, and CV transitions.** These carry the highest information density and are the least compressible in natural speech. The patent literature suggests even expanding CV transitions while compressing vowel steady-states.

4. **Maintain or increase the stressed/unstressed duration contrast.** Natural fast speakers do this automatically; artificial acceleration should preserve it.

5. **Respect phoneme-class-specific minimum durations.** The Klatt model's D_min concept is physiologically grounded and psychoacoustically validated.

### 6.2 Specific Algorithmic Guidance

The compression hierarchy for Osmium should approximately follow:

```
silence/pauses       >
unstressed vowels    >
stressed vowels      >
liquids/glides       >
nasals               >
fricative noise      >
stop closures        >
stop bursts/VOT/transitions (protect)
```

At moderate acceleration (1.3-1.5x), most savings come from pause compression and vowel shortening. At higher rates (1.5-2.0x), consonant compression begins but should be bounded by minimum duration floors. Beyond 2.0x, quality degrades for any method.

### 6.3 Recommended Datasets for Parameter Estimation

For extracting empirically grounded per-phoneme compression curves:

1. **BonnTempo** (primary): Matched same-speaker recordings at 5 rate levels across 5 languages.
2. **IFA Corpus** (supplementary): Phoneme-aligned Dutch speech in multiple speaking styles.
3. **Buckeye** (validation): Natural rate variation in American English conversation.
4. **BURNC** (supplementary): Professional speakers with prosodic annotations.
5. **LibriTTS** (scale): Large-scale audiobook data for data-driven modeling.

---

## 7. Key References

### Foundational Studies
- Crystal, T.H. & House, A.S. (1988). Segmental durations in connected-speech signals. JASA series.
- Gay, T. (1978). Effect of speaking rate on vowel formant movements. JASA, 63, 223-230.
- Klatt, D.H. (1979). Synthesis by rule of segmental durations in English sentences. In Lindblom & Ohman (Eds.), Frontiers of Speech Communication Research.
- Rosen, S. (1992). Temporal information in speech: acoustic, auditory and linguistic aspects. Phil. Trans. R. Soc. Lond. B, 336, 367-373.

### Speaking Rate and Duration
- Janse, E. (2004). Word perception in fast speech: artificially time-compressed vs. naturally produced fast speech. Speech Communication, 42, 155-173.
- van Santen, J.P.H. (1994). Assignment of segmental duration in text-to-speech synthesis. Computer Speech and Language, 8, 95-128.
- van Santen, J.P.H. (1993). Quantitative modeling of segmental duration. Proc. HLT 1993.
- Pfitzinger, H.R. (1999). Local speech rate perception in German speech. Proc. ICPhS 1999.
- Byrd, D. (1994). Relations of sex and dialect to reduction. Speech Communication, 15(1-2), 39-54.
- Fosler-Lussier, E. & Morgan, N. (1999). Effects of speaking rate and word frequency on pronunciations in conversational speech. Speech Communication, 29, 137-158.
- Quene, H. (2008). Multilevel modeling of between-speaker and within-speaker variation in spontaneous speech tempo. JASA, 123(2), 1104-1113.

### Natural Fast Speech
- Koreman, J. et al. (2016). Natural fast speech is perceived as faster than linearly time-compressed speech. Attention, Perception, & Psychophysics.
- Adank, P. & Janse, E. (2009). Perceptual learning of time-compressed and natural fast speech. JASA.
- Dellwo, V. et al. (2004). BonnTempo-Corpus & BonnTempo-Tools. Proc. Interspeech 2004.
- Pellegrino, F., Coupe, C. & Marsico, E. (2011). A cross-language perspective on speech information rate. Language, 87, 539-558.
- van Son, R.J.J.H. & Pols, L.C.W. (1999). An acoustic description of consonant reduction. Speech Communication, 28, 125-140.

### Time-Scale Modification
- Haque et al. (2023). Adaptive Time-Scale Modification for Improving Speech Intelligibility Based on Phoneme Clustering for Streaming Services. Proc. ICASSP 2023.
- STSM-FiLM (2025). A FiLM-Conditioned Neural Architecture for Time-Scale Modification of Speech. arXiv:2510.02672.
- US Patent 7,065,485 B1. Enhancing speech intelligibility using variable-rate time-scale modification.

### Corpora
- Anderson, A. et al. (1991). The HCRC Map Task Corpus. Language and Speech, 34(4).
- Pitt, M.A. et al. The Buckeye Corpus of Conversational Speech. Ohio State University.
- Ostendorf, M. et al. Boston University Radio Speech Corpus. LDC96S36.
- Panayotov, V. et al. (2015). Librispeech: An ASR Corpus Based on Public Domain Audio Books. ICASSP 2015.
- Garofolo, J.S. et al. (1993). DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus.
- van Son, R.J.J.H. et al. (2001). The IFA corpus. Proc. Eurospeech 2001.
