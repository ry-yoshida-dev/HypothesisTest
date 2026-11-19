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
class FlignerKilleen(MultiSampleTest):
    """
    Fligner-Killeen test of whether several populations have equal variances.

    Ranks the absolute deviations from each group's center, which makes it the
    most robust of the tests of equal variances to departures from normality.
    Undefined when every sample is constant.

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
        return HypothesisTestResult.from_outcome(
            stats.fligner(*samples, center=self.center.option_name),
            degrees_of_freedom=(len(samples) - 1,),
        )
