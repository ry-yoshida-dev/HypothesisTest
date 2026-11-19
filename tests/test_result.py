import math

import pytest
from scipy import stats

from hypothesis_test import ConfidenceInterval, HypothesisTestResult


def test_result_nan_statistic_is_undefined_and_clears_the_rest() -> None:
    result = HypothesisTestResult(
        statistic=math.nan,
        p_value=0.5,
        degrees_of_freedom=(3.0,),
        confidence_interval=ConfidenceInterval(low=0.0, high=1.0, level=0.95),
    )
    assert result == HypothesisTestResult(statistic=None)


def test_result_nan_p_value_is_undefined_but_keeps_the_statistic() -> None:
    result = HypothesisTestResult(statistic=2.0, p_value=math.nan)
    assert result.statistic == 2.0
    assert result.p_value is None


def test_result_p_value_is_clamped_to_one() -> None:
    assert HypothesisTestResult(statistic=0.0, p_value=1.0000000002).p_value == 1.0


@pytest.mark.parametrize("p_value", [-0.1, 1.5, math.inf])
def test_result_rejects_p_values_outside_the_unit_interval(p_value: float) -> None:
    with pytest.raises(ValueError, match="p_value"):
        HypothesisTestResult(statistic=1.0, p_value=p_value)


def test_result_degrees_of_freedom_are_floats() -> None:
    result = HypothesisTestResult(statistic=1.0, degrees_of_freedom=(2, 10))
    assert result.degrees_of_freedom == (2.0, 10.0)
    assert all(isinstance(value, float) for value in result.degrees_of_freedom)


def test_result_from_outcome_copies_statistic_and_p_value() -> None:
    outcome = stats.ttest_1samp([1.0, 2.0, 4.0], 0.0)
    result = HypothesisTestResult.from_outcome(outcome, degrees_of_freedom=(2,))
    assert result.statistic == pytest.approx(float(outcome.statistic))
    assert result.p_value == pytest.approx(float(outcome.pvalue))
    assert result.degrees_of_freedom == (2.0,)


@pytest.mark.parametrize(
    ("p_value", "level", "expected"),
    [(0.05, 0.05, True), (0.0500001, 0.05, False), (0.009, 0.01, True)],
)
def test_result_is_significant_includes_the_level(
    p_value: float, level: float, expected: bool
) -> None:
    assert HypothesisTestResult(statistic=1.0, p_value=p_value).is_significant(level) is expected


def test_result_is_significant_is_false_without_a_p_value() -> None:
    assert not HypothesisTestResult(statistic=None).is_significant()


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, math.nan])
def test_result_is_significant_rejects_levels_outside_the_unit_interval(level: float) -> None:
    with pytest.raises(ValueError, match="significance_level"):
        HypothesisTestResult(statistic=1.0, p_value=0.5).is_significant(level)


def test_result_is_significant_rejects_booleans() -> None:
    with pytest.raises(TypeError, match="significance_level"):
        HypothesisTestResult(statistic=1.0, p_value=0.5).is_significant(True)
