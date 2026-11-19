import warnings
from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..one_sample_test import OneSampleTest
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class ShapiroWilk(OneSampleTest):
    """
    Shapiro-Wilk test of whether a sample comes from a normal distribution.

    Among the most powerful normality tests for small and moderate samples. The
    statistic W is at most 1, with smaller values showing departure from normality.
    The p-value comes from an approximation that loses accuracy above 5000
    observations, where `DAgostinoPearson` or `JarqueBera` suit better.
    """

    min_samples = 3

    @override
    def _is_defined(self, sample: FloatArray) -> bool:
        return bool(np.ptp(sample) > 0)

    @override
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        with warnings.catch_warnings():
            # The docstring states the loss of accuracy that SciPy warns about.
            warnings.filterwarnings(
                "ignore", message="scipy.stats.shapiro: For N > 5000", category=UserWarning
            )
            return HypothesisTestResult.from_outcome(stats.shapiro(sample))
