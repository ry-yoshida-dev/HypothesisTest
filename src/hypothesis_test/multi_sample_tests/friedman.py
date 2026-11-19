from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..multi_sample_test import MultiSampleTest
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class Friedman(MultiSampleTest):
    """
    Friedman test of whether repeated measures of the same subjects come from one distribution.

    The nonparametric counterpart of repeated-measures analysis of variance. It
    ranks the observations within each subject, so it takes at least three
    measurements per subject. Undefined when every subject has equal measurements.
    """

    min_groups = 3
    is_repeated_measures = True

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        return bool(np.any(np.ptp(np.column_stack(samples), axis=1) > 0))

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(
            stats.friedmanchisquare(*samples), degrees_of_freedom=(len(samples) - 1,)
        )
