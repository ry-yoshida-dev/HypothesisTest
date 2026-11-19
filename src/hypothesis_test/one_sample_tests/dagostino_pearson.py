from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..array_types import FloatArray
from ..one_sample_test import OneSampleTest
from ..result import HypothesisTestResult


@dataclass(frozen=True)
class DAgostinoPearson(OneSampleTest):
    """
    D'Agostino-Pearson omnibus test of whether a sample comes from a normal distribution.

    Combines tests of skewness and kurtosis into a chi-square statistic with two
    degrees of freedom. Its kurtosis test needs at least 20 observations.
    """

    min_samples = 20

    @override
    def _is_defined(self, sample: FloatArray) -> bool:
        return bool(np.ptp(sample) > 0)

    @override
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(stats.normaltest(sample), degrees_of_freedom=(2,))
