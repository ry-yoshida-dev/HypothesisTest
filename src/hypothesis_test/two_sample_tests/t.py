import math
from dataclasses import dataclass
from typing import override

import numpy as np

from ..array_types import FloatArray
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..t_distribution import TDistribution
from ..two_sample_test import TwoSampleTest


@dataclass(frozen=True)
class TwoSampleT(TwoSampleTest):
    """
    Two-sample t-test of whether the difference between two population means equals a value.

    Welch's t-test by default, which stays valid when the variances differ; Student's
    t-test pools the variances instead. Both assume normally distributed
    observations, which matters less as the samples grow. The confidence interval is
    that of the mean of first minus the mean of second.

    Attributes
    ----------
    mean_difference : float
        The difference of the population means under the null hypothesis.
    is_variance_equal : bool
        Whether to assume equal variances and run Student's t-test.
    alternative : Alternative
        Alternative hypothesis about the difference of the population means.
    confidence_level : float
        Level of the confidence interval of the difference, in (0, 1).
    """

    mean_difference: float = 0.0
    is_variance_equal: bool = False
    alternative: Alternative = Alternative.TWO_SIDED
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        ParameterValidator.require_finite_numbers(mean_difference=self.mean_difference)
        ParameterValidator.require_booleans(is_variance_equal=self.is_variance_equal)
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_open_unit_interval(confidence_level=self.confidence_level)

    @override
    def _is_defined(self, first: FloatArray, second: FloatArray) -> bool:
        return bool(np.ptp(first) > 0 or np.ptp(second) > 0)

    @override
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        # SciPy's test has no hypothesized difference, and warns on a constant sample
        # that leaves the test well defined.
        first_size: int = first.size
        second_size: int = second.size
        first_variance: float = float(np.var(first, ddof=1))
        second_variance: float = float(np.var(second, ddof=1))
        standard_error: float
        degrees_of_freedom: float
        if self.is_variance_equal:
            degrees_of_freedom = first_size + second_size - 2
            pooled_variance: float = (
                (first_size - 1) * first_variance + (second_size - 1) * second_variance
            ) / degrees_of_freedom
            standard_error = math.sqrt(pooled_variance * (1 / first_size + 1 / second_size))
        else:
            first_share: float = first_variance / first_size
            second_share: float = second_variance / second_size
            standard_error = math.sqrt(first_share + second_share)
            degrees_of_freedom = standard_error**4 / (
                first_share**2 / (first_size - 1) + second_share**2 / (second_size - 1)
            )
        if not (math.isfinite(standard_error) and math.isfinite(degrees_of_freedom)):
            # The variances of extreme values overflow, leaving no reference distribution.
            return HypothesisTestResult(statistic=None)
        difference: float = float(np.mean(first)) - float(np.mean(second))
        statistic: float = (difference - self.mean_difference) / standard_error
        distribution: TDistribution = TDistribution(degrees_of_freedom)
        return HypothesisTestResult(
            statistic=statistic,
            p_value=distribution.p_value(statistic, self.alternative),
            degrees_of_freedom=(degrees_of_freedom,),
            confidence_interval=distribution.confidence_interval(
                difference, standard_error, self.alternative, self.confidence_level
            ),
        )
