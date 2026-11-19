from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..two_sample_test import TwoSampleTest
from ..types import FloatArray


@dataclass(frozen=True)
class BrunnerMunzel(TwoSampleTest):
    """
    Brunner-Munzel test of whether a value from one sample is as likely to be larger as smaller.

    Unlike the Mann-Whitney U test, it stays valid when the distributions differ in
    shape or spread. The statistic is positive when first tends to be larger.
    Undefined when, in both samples, every observation exceeds the same number of
    observations of the other sample, such as when the samples do not overlap.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis; GREATER means that first tends to be larger.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _is_defined(self, first: FloatArray, second: FloatArray) -> bool:
        # The variance estimate is built from these placements, so the statistic
        # has no reference distribution when neither sample's placements vary.
        pooled_ranks: FloatArray = np.asarray(stats.rankdata(np.concatenate([first, second])))
        first_placements: FloatArray = pooled_ranks[: first.size] - stats.rankdata(first)
        second_placements: FloatArray = pooled_ranks[first.size :] - stats.rankdata(second)
        return bool(np.ptp(first_placements) > 0 or np.ptp(second_placements) > 0)

    @override
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        outcome = stats.brunnermunzel(first, second, alternative=self.alternative.option_name)
        # SciPy's statistic is positive when second tends to be larger. Subtracting
        # from zero keeps a zero statistic from turning into -0.0.
        return HypothesisTestResult(
            statistic=0.0 - float(outcome.statistic), p_value=float(outcome.pvalue)
        )
