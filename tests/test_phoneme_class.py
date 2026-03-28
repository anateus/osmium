import numpy as np
from osmium.analyzer.phoneme_class import (
    PHONEME_CLASS_FLOORS,
    LABEL_TO_CLASS,
    classify_frame,
)


def test_all_mms_fa_labels_are_mapped():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    for label in labels:
        if label == "-":
            continue
        assert label in LABEL_TO_CLASS, f"Label '{label}' not mapped"


def test_plosive_labels():
    for label in "tdkgbpc'":
        assert LABEL_TO_CLASS[label] == "plosive", f"'{label}' should be plosive"


def test_fricative_labels():
    for label in "szfvhx":
        assert LABEL_TO_CLASS[label] == "fricative", f"'{label}' should be fricative"


def test_nasal_labels():
    for label in "mn":
        assert LABEL_TO_CLASS[label] == "nasal"


def test_liquid_glide_labels():
    for label in "lrwyj":
        assert LABEL_TO_CLASS[label] == "liquid_glide"


def test_vowel_labels():
    for label in "aeiou":
        assert LABEL_TO_CLASS[label] == "vowel"


def test_fallback_labels_are_plosive():
    for label in "q*":
        assert LABEL_TO_CLASS[label] == "plosive"


def test_floor_values_are_ordered():
    floors = PHONEME_CLASS_FLOORS
    assert floors["plosive"] > floors["fricative"]
    assert floors["fricative"] > floors["nasal"]
    assert floors["nasal"] > floors["liquid_glide"]
    assert floors["liquid_glide"] > floors["vowel"]
    assert floors["vowel"] > floors["silence"]


def test_classify_frame_silence():
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.9)
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "silence"


def test_classify_frame_plosive():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    t_idx = labels.index("t")
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.1)
    log_probs[t_idx] = np.log(0.7)
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "plosive"


def test_classify_frame_vowel():
    labels = list("-aienuotsrmkldghybpwcvjzf'qx*")
    a_idx = labels.index("a")
    n_labels = 29
    log_probs = np.full(n_labels, -10.0, dtype=np.float32)
    log_probs[0] = np.log(0.1)
    log_probs[a_idx] = np.log(0.6)
    cls = classify_frame(log_probs, blank_threshold=0.8)
    assert cls == "vowel"
