from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..array_types import FloatArray
from ..one_sample_test import OneSampleTest
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult


@dataclass(frozen=True)
class Sign(OneSampleTest):
    """
    Sign test of whether the population median equals a value.

    Makes no assumption about the shape of the distribution, at the cost of power.
    Observations equal to the hypothesized median are dropped. The statistic is the
    number of observations above the hypothesized median.

    Attributes
    ----------
    population_median : float
        The population median under the null hypothesis.
    alternative : Alternative
        Alternative hypothesis about the population median.
    """

    population_median: float = 0.0
    alternative: Alternative = Alternative.TWO_SIDED

    min_samples = 1

    def __post_init__(self) -> None:
        ParameterValidator.require_finite_numbers(population_median=self.population_median)
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _is_defined(self, sample: FloatArray) -> bool:
        return bool(np.any(sample != self.population_median))

    @override
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        num_above: int = int(np.count_nonzero(sample > self.population_median))
        num_below: int = int(np.count_nonzero(sample < self.population_median))
        outcome = stats.binomtest(
            num_above, num_above + num_below, alternative=self.alternative.option_name
        )
        return HypothesisTestResult(statistic=float(num_above), p_value=float(outcome.pvalue))
