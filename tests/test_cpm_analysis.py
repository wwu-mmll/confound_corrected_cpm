import numpy as np
import pytest


def test_create_permuted_y_structure(cpm_instance):
    """
    Test structural integrity: shape and value conservation.
    """
    # Use a larger y to prevent accidental collision of permutations
    y = np.arange(20)

    # Ensure the test runs with enough permutations to be meaningful
    # but not so many that it slows down (e.g., 50-100)
    if cpm_instance.n_permutations < 2:
        pytest.skip("Need at least 2 permutations to test variance")

    original_y_copy = y.copy()
    permuted_y = cpm_instance._create_permuted_y(y)

    # 1. Shape Check
    assert permuted_y.shape == (len(y), cpm_instance.n_permutations)

    # 2. Conservation Check (content is preserved)
    # Check a few random columns to save time, or all if fast
    for i in range(min(cpm_instance.n_permutations, 5)):
        assert sorted(permuted_y[:, i]) == list(y)

    # 3. Immutability Check (Original y must not change)
    np.testing.assert_array_equal(y, original_y_copy,
                                  err_msg="The function modified the original input array in place!")


def test_permutations_are_shuffled(cpm_instance):
    """
    Test that the output is actually randomized and not just repeated.
    """
    y = np.arange(50)
    permuted_y = cpm_instance._create_permuted_y(y)

    # 1. Check against the "Repeat" bug
    # Ensure column 0 is not identical to column 1
    # (Extremely unlikely to happen by chance with len=50)
    assert not np.array_equal(permuted_y[:, 0], permuted_y[:, 1]), \
        "Columns are identical! The shuffle likely failed (Repeat Bug)."

    # 2. Check against the "No-Op" bug
    # Ensure the first permutation is not identical to the original input
    assert not np.array_equal(permuted_y[:, 0], y), \
        "The permuted vector is identical to the input! (No shuffle occurred)"