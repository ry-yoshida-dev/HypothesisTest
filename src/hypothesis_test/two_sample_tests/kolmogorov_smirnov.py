from dataclasses import dataclass
from typing import override

from scipy import stats

from ..array_types import FloatArray
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..two_sample_test import TwoSampleTest


@dataclass(frozen=True)
class KolmogorovSmirnov(TwoSampleTest):
    """
    Two-sample Kolmogorov-Smirnov test of whether two samples come from one distribution.

    Detects any difference between the distributions, and is most sensitive near
    their medians. The statistic is the largest distance between the empirical
    distribution functions, in the direction of the alternative.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis; GREATER means that first tends to be larger. Unlike
        SciPy's ks_2samp, this refers to the values, not to the distribution
        functions.
    """

    alternative: Alternative = Alternative.TWO_SIDED

    min_samples = 1

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        # Larger values lower the distribution function, so SciPy takes the
        # alternative of second relative to first.
        return HypothesisTestResult.from_outcome(
            stats.ks_2samp(second, first, alternative=self.alternative.option_name)
        )
