import math

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy import stats

from hypothesis_test import (
    Alternative,
    DAgostinoPearson,
    HypothesisTestResult,
    JarqueBera,
    OneSampleT,
    OneSampleTest,
    ShapiroWilk,
    Sign,
    WilcoxonSignedRank,
)

SAMPLE: list[float] = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.4, 4.7, 5.9]
BEFORE: list[float] = [72.0, 68.0, 75.0, 80.0, 66.0, 71.0, 77.0, 69.0]
AFTER: list[float] = [70.0, 65.0, 74.0, 76.0, 66.5, 68.0, 75.0, 66.0]
UNDEFINED = HypothesisTestResult(statistic=None)

ALL_TESTS: list[OneSampleTest] = [
    OneSampleT(),
    WilcoxonSignedRank(),
    Sign(),
    ShapiroWilk(),
    DAgostinoPearson(),
    JarqueBera(),
]


@pytest.mark.parametrize("alternative", list(Alternative))
def test_one_sample_t_matches_the_formula_and_scipy(alternative: Alternative) -> None:
    sample = np.array(SAMPLE)
    mean: float = float(np.mean(sample))
    standard_error: float = float(np.std(sample, ddof=1)) / math.sqrt(sample.size)
    result = OneSampleT(population_mean=5.0, alternative=alternative, confidence_level=0.9).run(
        sample
    )
    reference = stats.ttest_1samp(sample, 5.0, alternative=alternative.option_name)
    interval = reference.confidence_interval(confidence_level=0.9)
    assert result.statistic == pytest.approx((mean - 5.0) / standard_error)
    assert result.p_value == pytest.approx(float(reference.pvalue))
    assert result.degrees_of_freedom == (sample.size - 1,)
    assert result.confidence_interval is not None
    assert result.confidence_interval.low == pytest.approx(float(interval.low))
    assert result.confidence_interval.high == pytest.approx(float(interval.high))
    assert result.confidence_interval.level == 0.9


def test_one_sample_t_paired_matches_the_paired_t_test() -> None:
    result = OneSampleT().run_paired(AFTER, BEFORE)
    reference = stats.ttest_rel(AFTER, BEFORE)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))


def test_wilcoxon_signed_rank_shifts_by_the_median() -> None:
    result = WilcoxonSignedRank(population_median=5.0).run(SAMPLE)
    reference = stats.wilcoxon(np.array(SAMPLE) - 5.0)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))


def test_wilcoxon_signed_rank_greater_sums_the_positive_ranks() -> None:
    differences = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
    ranks = stats.rankdata(np.abs(differences))
    result = WilcoxonSignedRank(alternative=Alternative.GREATER).run(differences)
    assert result.statistic == pytest.approx(float(np.sum(ranks[differences > 0])))


def test_wilcoxon_signed_rank_paired_matches_scipy() -> None:
    result = WilcoxonSignedRank().run_paired(AFTER, BEFORE)
    assert result.p_value == pytest.approx(float(stats.wilcoxon(AFTER, BEFORE).pvalue))


def test_sign_matches_the_binomial_distribution() -> None:
    # Eight above and two below the median; the two values on it are dropped.
    sample = [1.0] * 8 + [-1.0] * 2 + [0.0] * 2
    result = Sign().run(sample)
    assert result.statistic == 8.0
    assert result.p_value == pytest.approx(2 * (1 + 10 + 45) / 2**10)


def test_sign_greater_detects_a_larger_median() -> None:
    sample = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.5]
    assert Sign(population_median=1.0, alternative=Alternative.GREATER).run(
        sample
    ).p_value == pytest.approx(0.5**7)


def test_shapiro_wilk_matches_scipy() -> None:
    result = ShapiroWilk().run(SAMPLE)
    reference = stats.shapiro(SAMPLE)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))


def test_normality_tests_reject_a_skewed_sample() -> None:
    sample = np.random.default_rng(0).exponential(size=200)
    for test in [ShapiroWilk(), DAgostinoPearson(), JarqueBera()]:
        assert test.run(sample).is_significant(0.001)


def test_jarque_bera_matches_the_formula() -> None:
    sample = np.array(SAMPLE)
    deviations = sample - np.mean(sample)
    variance: float = float(np.mean(deviations**2))
    skewness: float = float(np.mean(deviations**3)) / variance**1.5
    kurtosis: float = float(np.mean(deviations**4)) / variance**2
    result = JarqueBera().run(sample)
    assert result.statistic == pytest.approx(
        sample.size / 6 * (skewness**2 + (kurtosis - 3) ** 2 / 4)
    )
    assert result.degrees_of_freedom == (2.0,)


def test_dagostino_pearson_needs_twenty_observations() -> None:
    sample = np.random.default_rng(2).normal(size=20)
    assert DAgostinoPearson().run(sample[:19]) == UNDEFINED
    result = DAgostinoPearson().run(sample)
    assert result.p_value == pytest.approx(float(stats.normaltest(sample).pvalue))


@pytest.mark.parametrize("test", ALL_TESTS)
def test_one_sample_tests_are_undefined_for_a_sample_without_spread(test: OneSampleTest) -> None:
    assert test.run([0.0] * 30) == UNDEFINED


@pytest.mark.parametrize("test", [OneSampleT(), ShapiroWilk(), JarqueBera()])
def test_one_sample_tests_are_undefined_for_too_few_observations(test: OneSampleTest) -> None:
    assert test.run([1.0]) == UNDEFINED


def test_one_sample_location_tests_are_defined_for_a_constant_off_the_median() -> None:
    assert Sign().run([2.0, 2.0, 2.0]).p_value == pytest.approx(0.25)
    assert WilcoxonSignedRank().run([2.0, 2.0, 2.0]).statistic is not None


@pytest.mark.parametrize("test", ALL_TESTS)
def test_one_sample_tests_run_without_warnings_on_random_data(test: OneSampleTest) -> None:
    result = test.run(np.random.default_rng(3).normal(size=40))
    assert result.statistic is not None
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize(
    ("sample", "error", "match"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], ValueError, "1-D"),
        ([], ValueError, "empty"),
        ([1.0, math.nan, 2.0], ValueError, "finite"),
        ([1.0, math.inf], ValueError, "finite"),
        (["a", "b"], TypeError, "numeric"),
        ([True, False], TypeError, "numeric"),
    ],
)
def test_one_sample_tests_reject_invalid_samples(
    sample: ArrayLike, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        OneSampleT().run(sample)


def test_one_sample_tests_reject_misaligned_pairs() -> None:
    with pytest.raises(ValueError, match="same length"):
        OneSampleT().run_paired([1.0, 2.0, 3.0], [1.0, 2.0])


def test_one_sample_tests_reject_invalid_options() -> None:
    with pytest.raises(ValueError, match="population_mean"):
        OneSampleT(population_mean=math.nan)
    with pytest.raises(ValueError, match="confidence_level"):
        OneSampleT(confidence_level=1.0)
    with pytest.raises(TypeError, match="alternative"):
        WilcoxonSignedRank(alternative="less")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="population_median"):
        Sign(population_median=True)
