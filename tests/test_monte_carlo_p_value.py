import numpy as np
import pytest

from hypothesis_test import Alternative, MonteCarloPValue


def test_monte_carlo_p_value_counts_extremes_per_observation() -> None:
    greater, less = MonteCarloPValue.count_extremes(
        [2.0, 10.0], [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    )
    # Ties count on both sides.
    assert greater.tolist() == [3, 0]
    assert less.tolist() == [2, 4]


def test_monte_carlo_p_value_counts_ties_within_a_relative_tolerance() -> None:
    observed: float = 1e6
    tied: float = observed * (1 + 1e-12)
    greater, less = MonteCarloPValue.count_extremes([observed], [[tied, tied]])
    assert greater.tolist() == [2]
    assert less.tolist() == [2]
    greater, less = MonteCarloPValue.count_extremes([observed], [[observed * (1 + 1e-6)]])
    assert (greater.tolist(), less.tolist()) == ([1], [0])


def test_monte_carlo_p_value_blockwise_counts_equal_one_shot_counts() -> None:
    rng = np.random.default_rng(0)
    observed = rng.normal(size=7)
    simulated = rng.normal(size=(7, 100))
    expected = MonteCarloPValue.count_extremes(observed, simulated)
    greater = np.zeros(7, dtype=np.int64)
    less = np.zeros(7, dtype=np.int64)
    for start in range(0, 100, 30):
        block_greater, block_less = MonteCarloPValue.count_extremes(
            observed, simulated[:, start : start + 30]
        )
        greater += block_greater
        less += block_less
    np.testing.assert_array_equal(greater, expected[0])
    np.testing.assert_array_equal(less, expected[1])


def test_monte_carlo_p_value_from_counts_adds_the_observed_statistic() -> None:
    greater = MonteCarloPValue.from_counts([0, 9], [99, 99], 99, Alternative.GREATER)
    np.testing.assert_allclose(greater, [1 / 100, 10 / 100])
    less = MonteCarloPValue.from_counts([0, 9], [99, 4], 99, Alternative.LESS)
    np.testing.assert_allclose(less, [1.0, 5 / 100])


def test_monte_carlo_p_value_two_sided_doubles_the_smaller_side_and_caps_at_one() -> None:
    np.testing.assert_allclose(
        MonteCarloPValue.from_counts([4, 60], [95, 60], 99, Alternative.TWO_SIDED),
        [2 * 5 / 100, 1.0],
    )


def test_monte_carlo_p_value_smallest_attainable_value_is_one_over_simulations_plus_one() -> None:
    p_value = MonteCarloPValue.from_simulations(100.0, np.arange(99.0), Alternative.GREATER)
    assert p_value == pytest.approx(1 / 100)
    assert MonteCarloPValue.from_simulations(
        100.0, np.arange(99.0), Alternative.TWO_SIDED
    ) == pytest.approx(2 / 100)


def test_monte_carlo_p_value_from_simulations_follows_the_alternative() -> None:
    simulated = np.arange(1.0, 10.0)
    assert MonteCarloPValue.from_simulations(8.5, simulated, Alternative.GREATER) == pytest.approx(
        2 / 10
    )
    assert MonteCarloPValue.from_simulations(8.5, simulated, Alternative.LESS) == pytest.approx(
        9 / 10
    )
    assert MonteCarloPValue.from_simulations(
        8.5, simulated, Alternative.TWO_SIDED
    ) == pytest.approx(4 / 10)


def test_monte_carlo_p_value_from_simulations_matches_counting_then_converting() -> None:
    rng = np.random.default_rng(1)
    simulated = rng.normal(size=200)
    greater, less = MonteCarloPValue.count_extremes([0.3], simulated[None, :])
    expected = MonteCarloPValue.from_counts(greater, less, 200, Alternative.TWO_SIDED)
    assert MonteCarloPValue.from_simulations(0.3, simulated, Alternative.TWO_SIDED) == expected[0]


def test_monte_carlo_p_value_count_extremes_rejects_invalid_input() -> None:
    with pytest.raises(TypeError, match="numeric"):
        MonteCarloPValue.count_extremes(["a"], [[1.0]])
    with pytest.raises(ValueError, match="1-D"):
        MonteCarloPValue.count_extremes([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="shape"):
        MonteCarloPValue.count_extremes([1.0, 2.0], [[1.0]])
    with pytest.raises(ValueError, match="shape"):
        MonteCarloPValue.count_extremes([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        MonteCarloPValue.count_extremes([np.nan], [[1.0]])
    with pytest.raises(ValueError, match="finite"):
        MonteCarloPValue.count_extremes([np.inf], [[np.inf, 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        MonteCarloPValue.count_extremes([1.0], [[1.0, np.nan]])


def test_monte_carlo_p_value_counts_infinite_simulated_statistics_as_extreme() -> None:
    greater, less = MonteCarloPValue.count_extremes([1.0], [[np.inf, -np.inf, 0.0]])
    assert (greater.tolist(), less.tolist()) == ([1], [2])


def test_monte_carlo_p_value_from_counts_accepts_empty_counts() -> None:
    p_values = MonteCarloPValue.from_counts([], [], 9, Alternative.TWO_SIDED)
    assert p_values.shape == (0,)


def test_monte_carlo_p_value_from_counts_rejects_invalid_input() -> None:
    two_sided = Alternative.TWO_SIDED
    with pytest.raises(TypeError, match="simulations"):
        MonteCarloPValue.from_counts([1], [1], True, two_sided)
    with pytest.raises(TypeError, match="simulations"):
        MonteCarloPValue.from_counts([1], [1], 9.0, two_sided)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 1"):
        MonteCarloPValue.from_counts([0], [0], 0, two_sided)
    with pytest.raises(TypeError, match="integers"):
        MonteCarloPValue.from_counts([1.0], [1], 9, two_sided)
    with pytest.raises(ValueError, match="1-D"):
        MonteCarloPValue.from_counts([[1]], [[1]], 9, two_sided)
    with pytest.raises(ValueError, match="same shape"):
        MonteCarloPValue.from_counts([1, 2], [1], 9, two_sided)
    with pytest.raises(ValueError, match="in \\[0, 9\\]"):
        MonteCarloPValue.from_counts([10], [1], 9, two_sided)
    with pytest.raises(ValueError, match="in \\[0, 9\\]"):
        MonteCarloPValue.from_counts([1], [-1], 9, two_sided)
    with pytest.raises(TypeError, match="Alternative"):
        MonteCarloPValue.from_counts([1], [1], 9, "less")  # type: ignore[arg-type]


def test_monte_carlo_p_value_from_simulations_rejects_invalid_input() -> None:
    greater = Alternative.GREATER
    with pytest.raises(ValueError, match="finite"):
        MonteCarloPValue.from_simulations(np.nan, [1.0], greater)
    with pytest.raises(ValueError, match="NaN"):
        MonteCarloPValue.from_simulations(1.0, [1.0, np.nan], greater)
    with pytest.raises(ValueError, match="1-D"):
        MonteCarloPValue.from_simulations(1.0, [[1.0]], greater)
    with pytest.raises(ValueError, match="empty"):
        MonteCarloPValue.from_simulations(1.0, [], greater)
    with pytest.raises(TypeError, match="numeric"):
        MonteCarloPValue.from_simulations(1.0, ["a"], greater)
