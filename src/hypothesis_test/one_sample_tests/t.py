import math
from dataclasses import dataclass
from typing import override

import numpy as np

from ..array_types import FloatArray
from ..one_sample_test import OneSampleTest
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..t_distribution import TDistribution


@dataclass(frozen=True)
class OneSampleT(OneSampleTest):
    """
    One-sample t-test of whether the population mean equals a value.

    Assumes normally distributed observations, which matters less as the sample grows.
    On paired data it is the paired t-test, and the confidence interval is that of
    the mean difference.

    Attributes
    ----------
    population_mean : float
        The population mean under the null hypothesis.
    alternative : Alternative
        Alternative hypothesis about the population mean.
    confidence_level : float
        Level of the confidence interval of the population mean, in (0, 1).
    """

    population_mean: float = 0.0
    alternative: Alternative = Alternative.TWO_SIDED
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        ParameterValidator.require_finite_numbers(population_mean=self.population_mean)
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_open_unit_interval(confidence_level=self.confidence_level)

    @override
    def _is_defined(self, sample: FloatArray) -> bool:
        return bool(np.ptp(sample) > 0)

    @override
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        mean: float = float(np.mean(sample))
        standard_error: float = float(np.std(sample, ddof=1)) / math.sqrt(sample.size)
        if not math.isfinite(standard_error):
            # The variance of extreme values overflows, leaving no reference distribution.
            return HypothesisTestResult(statistic=None)
        distribution: TDistribution = TDistribution(sample.size - 1)
        statistic: float = (mean - self.population_mean) / standard_error
        return HypothesisTestResult(
            statistic=statistic,
            p_value=distribution.p_value(statistic, self.alternative),
            degrees_of_freedom=(distribution.degrees_of_freedom,),
            confidence_interval=distribution.confidence_interval(
                mean, standard_error, self.alternative, self.confidence_level
            ),
        )
