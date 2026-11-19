from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..array_types import FloatArray
from ..multi_sample_test import MultiSampleTest
from ..result import HypothesisTestResult


@dataclass(frozen=True)
class Bartlett(MultiSampleTest):
    """
    Bartlett's test of whether several normal populations have equal variances.

    More powerful than Levene's test for normal data, but sensitive to departures
    from normality. Undefined when any sample is constant.
    """

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        return all(np.ptp(sample) > 0 for sample in samples)

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(
            stats.bartlett(*samples), degrees_of_freedom=(len(samples) - 1,)
        )
