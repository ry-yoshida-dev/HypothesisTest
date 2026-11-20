import math

import numpy as np
import pytest

from hypothesis_test import Alternative, NormalDistribution


def test_normal_distribution_p_value_matches_known_quantiles() -> None:
    assert NormalDistribution.p_value(1.96, Alternative.TWO_SIDED) == pytest.approx(0.05, abs=1e-3)
    assert NormalDistribution.p_value(0.0, Alternative.GREATER) == pytest.approx(0.5)
    assert NormalDistribution.p_value(1.6448536269514722, Alternative.GREATER) == pytest.approx(
        0.05
    )


def test_normal_distribution_p_value_follows_the_direction_of_the_alternative() -> None:
    assert NormalDistribution.p_value(2.0, Alternative.GREATER) < 0.05
    assert NormalDistribution.p_value(2.0, Alternative.LESS) > 0.95
    assert NormalDistribution.p_value(-2.0, Alternative.LESS) < 0.05
    assert NormalDistribution.p_value(-2.0, Alternative.TWO_SIDED) == pytest.approx(
        NormalDistribution.p_value(2.0, Alternative.TWO_SIDED)
    )


def test_normal_distribution_p_value_keeps_far_tails_accurate() -> None:
    expected: float = 0.5 * math.erfc(10 / math.sqrt(2))
    assert expected > 0
    assert NormalDistribution.p_value(10.0, Alternative.GREATER) == pytest.approx(expected)
    assert NormalDistribution.p_value(-10.0, Alternative.LESS) == pytest.approx(expected)
    assert NormalDistribution.p_value(10.0, Alternative.TWO_SIDED) == pytest.approx(2 * expected)


def test_normal_distribution_p_value_of_nan_is_nan() -> None:
    for alternative in Alternative:
        assert math.isnan(NormalDistribution.p_value(math.nan, alternative))


@pytest.mark.parametrize("alternative", list(Alternative))
def test_normal_distribution_p_values_match_the_scalar_p_value(alternative: Alternative) -> None:
    statistics: list[float] = [-3.0, -0.5, 0.0, 1.2, 8.0]
    np.testing.assert_allclose(
        NormalDistribution.p_values(statistics, alternative),
        [NormalDistribution.p_value(statistic, alternative) for statistic in statistics],
    )


def test_normal_distribution_p_values_keep_nan_in_place() -> None:
    result = NormalDistribution.p_values([1.0, np.nan, -1.0], Alternative.TWO_SIDED)
    assert np.isnan(result[1])
    assert not np.isnan(result[[0, 2]]).any()


def test_normal_distribution_p_values_accept_integers_and_empty_input() -> None:
    np.testing.assert_allclose(NormalDistribution.p_values([0], Alternative.LESS), [0.5])
    assert NormalDistribution.p_values([], Alternative.LESS).shape == (0,)


def test_normal_distribution_p_values_reject_invalid_input() -> None:
    with pytest.raises(TypeError, match="numeric"):
        NormalDistribution.p_values(["a"], Alternative.LESS)
    with pytest.raises(ValueError, match="1-D"):
        NormalDistribution.p_values([[1.0]], Alternative.LESS)
    with pytest.raises(TypeError, match="Alternative"):
        NormalDistribution.p_values([1.0], "less")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Alternative"):
        NormalDistribution.p_value(1.0, "less")  # type: ignore[arg-type]
