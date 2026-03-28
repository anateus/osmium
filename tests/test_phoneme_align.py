import numpy as np
from osmium.analyzer.phoneme_align import phoneme_segments_to_importance
from osmium.analyzer.importance import ImportanceMap
from osmium.analyzer.phoneme_class import PHONEME_CLASS_FLOORS


def test_segments_to_importance_single_vowel():
    segments = [("a", 0.0, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=50.0)
    assert isinstance(result, ImportanceMap)
    np.testing.assert_allclose(
        result.scores, PHONEME_CLASS_FLOORS["vowel"], atol=1e-6
    )


def test_segments_to_importance_plosive_vowel():
    segments = [("t", 0.0, 0.1), ("a", 0.1, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=100.0)
    assert result.scores[0] == PHONEME_CLASS_FLOORS["plosive"]
    assert result.scores[-1] == PHONEME_CLASS_FLOORS["vowel"]


def test_segments_to_importance_covers_full_duration():
    segments = [("s", 0.0, 0.5), ("a", 0.5, 1.0)]
    result = phoneme_segments_to_importance(segments, duration=1.0, frame_rate=50.0)
    assert len(result.scores) == 50
    assert result.duration == 1.0


def test_segments_to_importance_empty():
    result = phoneme_segments_to_importance([], duration=1.0, frame_rate=50.0)
    assert len(result.scores) == 50
    np.testing.assert_allclose(
        result.scores, PHONEME_CLASS_FLOORS["silence"], atol=1e-6
    )
