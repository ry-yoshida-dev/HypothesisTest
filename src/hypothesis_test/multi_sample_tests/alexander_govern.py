from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..multi_sample_test import MultiSampleTest
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class AlexanderGovern(MultiSampleTest):
    """
    Alexander-Govern test of whether several population means are equal.

    Like Welch's F-test, it allows unequal variances, and it keeps its size better
    when the variances and sample sizes are unbalanced. Assumes normally distributed
    observations. Undefined when any sample is constant.
    """

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        return all(np.ptp(sample) > 0 for sample in samples)

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(
            stats.alexandergovern(*samples), degrees_of_freedom=(len(samples) - 1,)
        )
