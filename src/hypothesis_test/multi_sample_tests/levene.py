from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..multi_sample_test import MultiSampleTest
from ..options import Center
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import FloatArray


@dataclass(frozen=True)
class Levene(MultiSampleTest):
    """
    Levene's test of whether several populations have equal variances.

    An F-test on the absolute deviations from each group's center, robust to
    departures from normality with the median as the center. Undefined when every
    sample is constant.

    Attributes
    ----------
    center : Center
        Center from which deviations are measured.
    """

    center: Center = Center.MEDIAN

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Center, center=self.center)

    @override
    def _is_defined(self, samples: list[FloatArray]) -> bool:
        return any(np.ptp(sample) > 0 for sample in samples)

    @override
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        num_observations: int = sum(sample.size for sample in samples)
        return HypothesisTestResult.from_outcome(
            stats.levene(*samples, center=self.center.option_name),
            degrees_of_freedom=(len(samples) - 1, num_observations - len(samples)),
        )
