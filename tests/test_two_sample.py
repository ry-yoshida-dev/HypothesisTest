import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from hypothesis_test import (
    Alternative,
    BrunnerMunzel,
    CramerVonMises,
    HypothesisTestResult,
    KolmogorovSmirnov,
    MannWhitneyU,
    TwoSampleT,
    TwoSampleTest,
)

FIRST: list[float] = [23.1, 25.4, 21.8, 26.0, 24.7, 22.9, 27.3, 25.1]
SECOND: list[float] = [20.2, 22.5, 19.8, 23.0, 21.1, 18.9, 24.2, 20.7, 22.0, 21.5]
UNDEFINED = HypothesisTestResult(statistic=None)

ALL_TESTS: list[TwoSampleTest] = [
    TwoSampleT(),
    TwoSampleT(is_variance_equal=True),
    MannWhitneyU(),
    BrunnerMunzel(),
    KolmogorovSmirnov(),
    CramerVonMises(),
]


@pytest.mark.parametrize("is_variance_equal", [True, False])
@pytest.mark.parametrize("alternative", list(Alternative))
def test_two_sample_t_matches_scipy(is_variance_equal: bool, alternative: Alternative) -> None:
    result = TwoSampleT(
        is_variance_equal=is_variance_equal, alternative=alternative, confidence_level=0.9
    ).run(FIRST, SECOND)
    reference = stats.ttest_ind(
        FIRST, SECOND, equal_var=is_variance_equal, alternative=alternative.option_name
    )
    interval = reference.confidence_interval(confidence_level=0.9)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))
    assert result.degrees_of_freedom == pytest.approx((float(reference.df),))
    assert result.confidence_interval is not None
    assert result.confidence_interval.low == pytest.approx(float(interval.low))
    assert result.confidence_interval.high == pytest.approx(float(interval.high))


def test_two_sample_t_tests_the_hypothesized_difference() -> None:
    shifted = np.array(SECOND) + 1.5
    assert TwoSampleT(mean_difference=-1.5).run(FIRST, shifted).statistic == pytest.approx(
        TwoSampleT().run(FIRST, SECOND).statistic
    )


def test_two_sample_t_welch_is_defined_with_one_constant_sample() -> None:
    result = TwoSampleT().run([5.0, 5.0, 5.0], SECOND)
    assert result.degrees_of_freedom == pytest.approx((len(SECOND) - 1,))
    assert result.statistic == pytest.approx(
        (5.0 - float(np.mean(SECOND))) / (float(np.std(SECOND, ddof=1)) / math.sqrt(len(SECOND)))
    )


def test_mann_whitney_u_counts_pairs_where_first_is_larger() -> None:
    first = np.array([1.0, 4.0, 4.0, 7.0])
    second = np.array([2.0, 4.0, 5.0])
    differences: NDArray[np.float64] = first[:, np.newaxis] - second[np.newaxis, :]
    expected: float = np.count_nonzero(differences > 0) + 0.5 * np.count_nonzero(differences == 0)
    assert MannWhitneyU().run(first, second).statistic == pytest.approx(expected)


def test_mann_whitney_u_is_exact_for_separated_samples_without_ties() -> None:
    # Only one of the C(6, 3) orderings puts all of first below second.
    result = MannWhitneyU(alternative=Alternative.LESS).run([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert result.statistic == 0.0
    assert result.p_value == pytest.approx(1 / 20)


def test_mann_whitney_u_is_undefined_when_all_values_tie() -> None:
    assert MannWhitneyU().run([1.0, 1.0], [1.0, 1.0, 1.0]) == UNDEFINED


def test_kolmogorov_smirnov_statistic_is_the_largest_gap() -> None:
    first = np.array(FIRST)
    second = np.array(SECOND)
    points = np.concatenate([first, second])
    first_cdf = np.searchsorted(np.sort(first), points, side="right") / first.size
    second_cdf = np.searchsorted(np.sort(second), points, side="right") / second.size
    result = KolmogorovSmirnov().run(first, second)
    assert result.statistic == pytest.approx(float(np.max(np.abs(first_cdf - second_cdf))))


@pytest.mark.parametrize(
    "test",
    [MannWhitneyU, BrunnerMunzel, KolmogorovSmirnov, TwoSampleT],
)
def test_two_sample_greater_means_that_first_is_larger(
    test: type[MannWhitneyU | BrunnerMunzel | KolmogorovSmirnov | TwoSampleT],
) -> None:
    greater = test(alternative=Alternative.GREATER).run(FIRST, SECOND).p_value
    less = test(alternative=Alternative.LESS).run(FIRST, SECOND).p_value
    assert greater is not None
    assert less is not None
    assert greater < 0.05 < less


def test_brunner_munzel_statistic_is_positive_when_first_is_larger() -> None:
    forward = BrunnerMunzel().run(FIRST, SECOND)
    backward = BrunnerMunzel().run(SECOND, FIRST)
    assert forward.statistic is not None
    assert forward.statistic > 0
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)


def test_brunner_munzel_is_undefined_for_separated_samples() -> None:
    assert BrunnerMunzel().run([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == UNDEFINED


def test_cramer_von_mises_matches_scipy() -> None:
    result = CramerVonMises().run(FIRST, SECOND)
    reference = stats.cramervonmises_2samp(FIRST, SECOND)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))


@pytest.mark.parametrize(
    "test", [TwoSampleT(), TwoSampleT(is_variance_equal=True), BrunnerMunzel()]
)
def test_two_sample_tests_are_undefined_for_constant_samples(test: TwoSampleTest) -> None:
    assert test.run([2.0, 2.0, 2.0], [2.0, 2.0]) == UNDEFINED


@pytest.mark.parametrize("test", [TwoSampleT(), BrunnerMunzel(), CramerVonMises()])
def test_two_sample_tests_are_undefined_for_too_few_observations(test: TwoSampleTest) -> None:
    assert test.run([1.0], SECOND) == UNDEFINED


@pytest.mark.parametrize("test", ALL_TESTS)
def test_two_sample_tests_run_without_warnings_on_random_data(test: TwoSampleTest) -> None:
    rng = np.random.default_rng(0)
    result = test.run(rng.normal(size=30), rng.normal(0.3, 2.0, size=25))
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("test", ALL_TESTS)
def test_two_sample_tests_run_without_warnings_on_extreme_values(test: TwoSampleTest) -> None:
    test.run([1e308, -1e308, 5e307], [0.0, 1.0, 2.0])


@pytest.mark.parametrize("is_variance_equal", [True, False])
def test_two_sample_t_is_undefined_when_the_variance_overflows(is_variance_equal: bool) -> None:
    result = TwoSampleT(is_variance_equal=is_variance_equal).run([1e308, -1e308, 5e307], [0.0, 1.0])
    assert result == UNDEFINED


def test_two_sample_tests_reject_invalid_samples_by_name() -> None:
    with pytest.raises(ValueError, match="second must be finite"):
        MannWhitneyU().run(FIRST, [1.0, math.nan])


def test_two_sample_tests_reject_invalid_options() -> None:
    with pytest.raises(TypeError, match="is_variance_equal"):
        TwoSampleT(is_variance_equal=1)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="mean_difference"):
        TwoSampleT(mean_difference=math.inf)
    with pytest.raises(TypeError, match="alternative"):
        KolmogorovSmirnov(alternative=None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
