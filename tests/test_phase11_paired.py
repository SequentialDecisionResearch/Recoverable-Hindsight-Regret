import numpy as np

from trading_policy_oracle_repair.inference import bootstrap_paired_path, path_delta_metrics_from_arrays


def test_paired_bootstrap_preserves_constant_daily_advantage_exactly():
    n = 63
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 0.01, n)
    advantage = 0.001
    arrays = {
        "reference_log_returns": ref,
        "adaptive_log_returns": ref + advantage,
        "reference_turnover": np.zeros(n),
        "adaptive_turnover": np.zeros(n),
        "reference_cost_fraction": np.zeros(n),
        "adaptive_cost_fraction": np.zeros(n),
    }
    obs = path_delta_metrics_from_arrays(arrays)
    assert abs(obs["log_utility_improvement"] - n * advantage) < 1e-12
    draws = bootstrap_paired_path(arrays, block_length=20, resamples=100, annualization_days=252, seed=4)
    assert np.allclose(draws["log_utility_improvement"], n * advantage)


def test_paired_bootstrap_is_deterministic_for_seed():
    n = 80
    ref = np.linspace(-0.01, 0.01, n)
    arrays = {
        "reference_log_returns": ref,
        "adaptive_log_returns": ref + np.sin(np.arange(n)) * 0.001,
        "reference_turnover": np.zeros(n),
        "adaptive_turnover": np.ones(n) * 0.01,
        "reference_cost_fraction": np.zeros(n),
        "adaptive_cost_fraction": np.ones(n) * 0.0001,
    }
    a = bootstrap_paired_path(arrays, block_length=20, resamples=50, annualization_days=252, seed=99)
    b = bootstrap_paired_path(arrays, block_length=20, resamples=50, annualization_days=252, seed=99)
    for k in a:
        assert np.array_equal(a[k], b[k])
