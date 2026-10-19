import numpy as np

from trading_policy_oracle_repair.inference import circular_block_bootstrap_indices


def test_circular_blocks_are_consecutive_modulo_n():
    rng = np.random.default_rng(3)
    idx = circular_block_bootstrap_indices(11, block_length=4, resamples=7, rng=rng)
    assert idx.shape == (7, 11)
    assert idx.min() >= 0 and idx.max() < 11
    # The first two complete blocks in every draw are consecutive modulo n.
    for row in idx:
        for start in (0, 4):
            block = row[start:start+4]
            assert np.all(block[1:] == (block[:-1] + 1) % 11)
