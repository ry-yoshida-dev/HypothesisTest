from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..array_types import FloatArray
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..two_sample_test import TwoSampleTest


@dataclass(frozen=True)
class MannWhitneyU(TwoSampleTest):
    """
    Mann-Whitney U test of whether one sample tends to take larger values than the other.

    The nonparametric counterpart of the two-sample t-test, also called the Wilcoxon
    rank-sum test. Assumes that the distributions have the same shape under the null
    hypothesis. The statistic is the U statistic of first. The p-value is exact for
    small samples without ties. Undefined when all observations are equal.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis; GREATER means that first tends to be larger.
    is_continuity_corrected : bool
        Whether to apply the continuity correction to the normal approximation.
    """

    alternative: Alternative = Alternative.TWO_SIDED
    is_continuity_corrected: bool = True

    min_samples = 1

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_booleans(is_continuity_corrected=self.is_continuity_corrected)

    @override
    def _is_defined(self, first: FloatArray, second: FloatArray) -> bool:
        return bool(np.ptp(np.concatenate([first, second])) > 0)

    @override
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(
            stats.mannwhitneyu(
                first,
                second,
                use_continuity=self.is_continuity_corrected,
                alternative=self.alternative.option_name,
            )
        )
