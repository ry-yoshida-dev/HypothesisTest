from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..array_types import FloatArray
from ..multi_sample_test import MultiSampleTest
from ..result import HypothesisTestResult


@dataclass(frozen=True)
class KruskalWallis(MultiSampleTest):
    """
    Kruskal-Wallis H test of whether several samples come from one distribution.

    The nonparametric counterpart of one-way analysis of variance, sensitive mainly
    to differences in location. Undefined when all observations are equal.
    """

    min_samples = 1

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        return bool(np.ptp(np.concatenate(samples)) > 0)

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(
            stats.kruskal(*samples), degrees_of_freedom=(len(samples) - 1,)
        )
