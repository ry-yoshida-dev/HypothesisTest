from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..one_sample_test import OneSampleTest
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class WilcoxonSignedRank(OneSampleTest):
    """
    Wilcoxon signed-rank test of whether a symmetric distribution is centered at a value.

    The nonparametric counterpart of the one-sample and paired t-tests. Observations
    equal to the hypothesized median are dropped. The statistic is the smaller of
    the signed-rank sums for a two-sided alternative, and the sum of the ranks of
    positive deviations otherwise.

    Attributes
    ----------
    population_median : float
        The center of the distribution under the null hypothesis.
    alternative : Alternative
        Alternative hypothesis about the center.
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
        return HypothesisTestResult.from_outcome(
            stats.wilcoxon(
                sample - self.population_median, alternative=self.alternative.option_name
            )
        )
