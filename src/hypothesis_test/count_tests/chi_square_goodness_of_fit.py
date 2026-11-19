import math
from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..count_test import CountTest
from ..options import DivergenceStatistic
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import FloatArray, IntArray


@dataclass(frozen=True)
class ChiSquareGoodnessOfFit(CountTest):
    """
    Chi-square goodness-of-fit test of whether categories occur with given proportions.

    Takes the frequency of each category. The chi-square approximation needs
    expected frequencies of about five or more.

    Attributes
    ----------
    expected_proportions : tuple[float, ...] | None
        Proportion of each category under the null hypothesis, positive and summing to
        1; None for equal proportions.
    statistic : DivergenceStatistic
        Statistic that compares observed and expected frequencies.
    """

    expected_proportions: tuple[float, ...] | None = None
    statistic: DivergenceStatistic = DivergenceStatistic.PEARSON

    shape = (None,)

    def __post_init__(self) -> None:
        ParameterValidator.require_members(DivergenceStatistic, statistic=self.statistic)
        if self.expected_proportions is None:
            return
        object.__setattr__(self, "expected_proportions", tuple(self.expected_proportions))
        ParameterValidator.require_open_unit_interval(
            **{
                f"expected_proportions[{index}]": value
                for index, value in enumerate(self.expected_proportions)
            }
        )
        if not math.isclose(math.fsum(self.expected_proportions), 1.0, abs_tol=1e-9):
            raise ValueError(
                f"expected_proportions must sum to 1, got {math.fsum(self.expected_proportions)}"
            )

    @override
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        proportions: FloatArray = (
            np.full(counts.size, 1 / counts.size)
            if self.expected_proportions is None
            else self._matching_proportions(counts.size)
        )
        expected: FloatArray = proportions / np.sum(proportions) * np.sum(counts)
        return HypothesisTestResult.from_outcome(
            stats.power_divergence(counts, expected, lambda_=self.statistic.option_name),
            degrees_of_freedom=(counts.size - 1,),
        )

    def _matching_proportions(self, num_categories: int) -> FloatArray:
        proportions: FloatArray = np.array(self.expected_proportions, dtype=np.float64)
        if proportions.size != num_categories:
            raise ValueError(
                f"counts has {num_categories} categories, but expected_proportions has {proportions.size}"
            )
        return proportions
