import numpy as np
from osmium.tsm.rate_schedule import importance_to_rate_schedule


def test_gamma_1_produces_linear_rates():
    """gamma=1.0 should produce rates using the linear inv_importance formula."""
    importance = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    times = np.linspace(0, 1.0, 5)
    rates, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.0,
        smoothing_sigma=0, max_rate_change=100.0,
    )
    assert rates[0] > rates[2] > rates[4]
    diffs = np.diff(rates)
    assert np.all(diffs < 0)


def test_gamma_compresses_mid_range_rates():
    """gamma=1.5 should produce lower rates for mid-importance content vs gamma=1.0."""
    importance = np.linspace(0, 1, 50)
    times = np.linspace(0, 5.0, 50)
    rates_linear, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.0,
    )
    rates_compressed, _ = importance_to_rate_schedule(
        importance, times, target_speed=3.0, gamma=1.5,
    )
    mid = len(importance) // 2
    mid_slice = slice(mid - 3, mid + 3)
    assert rates_compressed[mid_slice].mean() < rates_linear[mid_slice].mean()


def test_gamma_preserves_target_speed():
    """Total output duration should still match target speed regardless of gamma."""
    importance = np.random.RandomState(42).rand(200)
    times = np.linspace(0, 10.0, 200)
    for gamma in [1.0, 1.5, 2.0]:
        rates, _ = importance_to_rate_schedule(
            importance, times, target_speed=3.0, gamma=gamma,
        )
        dt = np.diff(times, prepend=0)
        dt[0] = times[0]
        output_dur = (dt / rates).sum()
        target_dur = times[-1] / 3.0
        assert abs(output_dur - target_dur) / target_dur < 0.01, (
            f"gamma={gamma}: output_dur={output_dur:.3f} vs target={target_dur:.3f}"
        )
