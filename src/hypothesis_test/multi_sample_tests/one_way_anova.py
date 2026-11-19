from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..multi_sample_test import MultiSampleTest
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class OneWayAnova(MultiSampleTest):
    """
    One-way analysis of variance of whether several population means are equal.

    Assumes normally distributed observations. Fisher's F-test assumes equal
    variances, and Welch's F-test does not. Undefined when every sample is constant,
    and for Welch's test when any sample is.

    Attributes
    ----------
    is_variance_equal : bool
        Whether to assume equal variances and run Fisher's F-test instead of Welch's.
    """

    is_variance_equal: bool = True

    def __post_init__(self) -> None:
        ParameterValidator.require_booleans(is_variance_equal=self.is_variance_equal)

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        spreads: FloatArray = np.array([np.ptp(sample) for sample in samples])
        if self.is_variance_equal:
            return bool(np.any(spreads > 0))
        return bool(np.all(spreads > 0))

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        num_groups: int = len(samples)
        sizes: FloatArray = np.array([sample.size for sample in samples], dtype=np.float64)
        denominator_degrees: float
        if self.is_variance_equal:
            denominator_degrees = float(np.sum(sizes)) - num_groups
        else:
            # Welch's denominator degrees of freedom, which SciPy does not report.
            variances: FloatArray = np.array([np.var(sample, ddof=1) for sample in samples])
            weights: FloatArray = sizes / variances
            spread: float = float(np.sum((1 - weights / np.sum(weights)) ** 2 / (sizes - 1)))
            denominator_degrees = (num_groups**2 - 1) / (3 * spread)
        return HypothesisTestResult.from_outcome(
            stats.f_oneway(*samples, equal_var=self.is_variance_equal),
            degrees_of_freedom=(num_groups - 1, denominator_degrees),
        )
