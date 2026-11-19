from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..one_sample_test import OneSampleTest
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class JarqueBera(OneSampleTest):
    """
    Jarque-Bera test of whether a sample comes from a normal distribution.

    Compares the sample skewness and kurtosis with those of a normal distribution
    through their asymptotic chi-square distribution with two degrees of freedom,
    so it suits large samples.
    """

    @override
    def _is_defined(self, sample: FloatArray) -> bool:
        return bool(np.ptp(sample) > 0)

    @override
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(stats.jarque_bera(sample), degrees_of_freedom=(2,))
