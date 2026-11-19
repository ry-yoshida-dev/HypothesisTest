import math

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from hypothesis_test import (
    Alternative,
    Binomial,
    ChiSquareGoodnessOfFit,
    ChiSquareIndependence,
    CountTest,
    DivergenceStatistic,
    FisherExact,
    HypothesisTestResult,
    McNemar,
    McNemarVariant,
)

OBSERVED: list[int] = [18, 25, 37]
TABLE: list[list[int]] = [[12, 7, 9], [5, 14, 11]]
TWO_BY_TWO: list[list[int]] = [[8, 2], [1, 5]]
UNDEFINED = HypothesisTestResult(statistic=None)


def test_goodness_of_fit_matches_the_pearson_formula() -> None:
    observed: NDArray[np.int64] = np.array(OBSERVED, dtype=np.int64)
    expected: NDArray[np.float64] = np.array([0.25, 0.25, 0.5]) * float(observed.sum())
    result = ChiSquareGoodnessOfFit(expected_proportions=(0.25, 0.25, 0.5)).run(observed)
    statistic: float = float(np.sum((observed - expected) ** 2 / expected))
    assert result.statistic == pytest.approx(statistic)
    assert result.p_value == pytest.approx(float(stats.chi2.sf(statistic, 2)))
    assert result.degrees_of_freedom == (2.0,)


def test_goodness_of_fit_defaults_to_equal_proportions_and_takes_the_g_statistic() -> None:
    observed = np.array(OBSERVED)
    expected: float = observed.sum() / observed.size
    result = ChiSquareGoodnessOfFit(statistic=DivergenceStatistic.LOG_LIKELIHOOD).run(observed)
    assert result.statistic == pytest.approx(
        2 * float(np.sum(observed * np.log(observed / expected)))
    )


def test_goodness_of_fit_rejects_invalid_proportions() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        ChiSquareGoodnessOfFit(expected_proportions=(0.5, 0.6))
    with pytest.raises(ValueError, match=r"expected_proportions\[1\]"):
        ChiSquareGoodnessOfFit(expected_proportions=(0.6, -0.1, 0.5))
    with pytest.raises(ValueError, match="3 categories"):
        ChiSquareGoodnessOfFit(expected_proportions=(0.5, 0.5)).run(OBSERVED)


def test_independence_matches_the_pearson_formula() -> None:
    observed = np.array(TABLE, dtype=np.float64)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    result = ChiSquareIndependence().run(TABLE)
    assert result.statistic == pytest.approx(float(np.sum((observed - expected) ** 2 / expected)))
    assert result.degrees_of_freedom == (2.0,)


def test_independence_applies_yates_correction_to_two_by_two_tables() -> None:
    observed = np.array([[20, 15], [10, 25]], dtype=np.float64)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    corrected: float = float(np.sum((np.abs(observed - expected) - 0.5) ** 2 / expected))
    uncorrected: float = float(np.sum((observed - expected) ** 2 / expected))
    assert ChiSquareIndependence().run(observed).statistic == pytest.approx(corrected)
    assert ChiSquareIndependence(is_corrected=False).run(observed).statistic == pytest.approx(
        uncorrected
    )


def test_independence_is_undefined_with_an_empty_row() -> None:
    assert ChiSquareIndependence().run([[0, 0], [3, 4]]) == UNDEFINED


def test_fisher_exact_greater_is_the_hypergeometric_upper_tail() -> None:
    # With the margins fixed, the top-left cell follows a hypergeometric distribution.
    table: NDArray[np.int64] = np.array(TWO_BY_TWO, dtype=np.int64)
    total: int = int(table.sum())
    row: int = int(table[0].sum())
    column: int = int(table[:, 0].sum())
    expected: float = float(stats.hypergeom.sf(int(table[0, 0]) - 1, total, row, column))
    result = FisherExact(alternative=Alternative.GREATER).run(table)
    assert result.p_value == pytest.approx(expected)


def test_fisher_exact_interval_contains_its_estimate() -> None:
    result = FisherExact(confidence_level=0.9).run(TWO_BY_TWO)
    assert result.statistic is not None
    assert result.confidence_interval is not None
    assert result.confidence_interval.low < result.statistic < result.confidence_interval.high
    assert result.confidence_interval.level == 0.9


def test_fisher_exact_is_undefined_with_an_empty_column() -> None:
    assert FisherExact().run([[0, 3], [0, 4]]) == UNDEFINED


def test_mcnemar_exact_is_the_binomial_test_of_discordant_pairs() -> None:
    result = McNemar().run([[30, 9], [2, 25]])
    assert result.statistic == 2.0
    assert result.p_value == pytest.approx(2 * (1 + 11 + 55) / 2**11)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [(McNemarVariant.ASYMPTOTIC, 7**2 / 11), (McNemarVariant.CORRECTED, 6**2 / 11)],
)
def test_mcnemar_chi_square_matches_the_formula(variant: McNemarVariant, expected: float) -> None:
    result = McNemar(variant=variant).run([[30, 9], [2, 25]])
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(float(stats.chi2.sf(expected, 1)))
    assert result.degrees_of_freedom == (1.0,)


def test_mcnemar_correction_never_overshoots_balanced_counts() -> None:
    assert McNemar(variant=McNemarVariant.CORRECTED).run([[5, 4], [4, 5]]).statistic == 0.0


def test_mcnemar_is_undefined_without_discordant_pairs() -> None:
    assert McNemar().run([[10, 0], [0, 12]]) == UNDEFINED


def test_binomial_matches_the_binomial_distribution() -> None:
    result = Binomial(expected_proportion=0.3, alternative=Alternative.GREATER).run([7, 3])
    assert result.statistic == pytest.approx(0.7)
    assert result.p_value == pytest.approx(float(stats.binom.sf(6, 10, 0.3)))


def test_binomial_interval_is_clopper_pearson() -> None:
    result = Binomial(confidence_level=0.9).run([7, 3])
    assert result.confidence_interval is not None
    assert result.confidence_interval.low == pytest.approx(float(stats.beta.ppf(0.05, 7, 4)))
    assert result.confidence_interval.high == pytest.approx(float(stats.beta.ppf(0.95, 8, 3)))


@pytest.mark.parametrize(
    ("test", "counts"),
    [
        (ChiSquareGoodnessOfFit(), [0, 0, 0]),
        (ChiSquareIndependence(), [[0, 0], [0, 0]]),
        (FisherExact(), [[0, 0], [0, 0]]),
        (Binomial(), [0, 0]),
    ],
)
def test_count_tests_are_undefined_without_observations(test: CountTest, counts: ArrayLike) -> None:
    assert test.run(counts) == UNDEFINED


@pytest.mark.parametrize(
    ("test", "counts"),
    [
        (ChiSquareGoodnessOfFit(), [5]),
        (ChiSquareGoodnessOfFit(), [[1, 2], [3, 4]]),
        (ChiSquareIndependence(), [[1, 2, 3]]),
        (FisherExact(), TABLE),
        (McNemar(), [1, 2, 3, 4]),
        (Binomial(), [1, 2, 3]),
    ],
)
def test_count_tests_reject_other_shapes(test: CountTest, counts: ArrayLike) -> None:
    with pytest.raises(ValueError, match="shape"):
        test.run(counts)


@pytest.mark.parametrize(
    ("counts", "error", "match"),
    [
        ([3, -1], ValueError, "negative"),
        ([3, 1.5], ValueError, "integer"),
        ([3, math.nan], ValueError, "finite"),
        (["3", "1"], TypeError, "numeric"),
    ],
)
def test_count_tests_reject_invalid_counts(
    counts: ArrayLike, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        Binomial().run(counts)


def test_count_tests_accept_integral_floats() -> None:
    assert Binomial().run([7.0, 3.0]) == Binomial().run([7, 3])


def test_count_tests_reject_invalid_options() -> None:
    with pytest.raises(ValueError, match="expected_proportion"):
        Binomial(expected_proportion=1.0)
    with pytest.raises(TypeError, match="variant"):
        McNemar(variant="exact")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="is_corrected"):
        ChiSquareIndependence(is_corrected=None)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
