from dataclasses import dataclass
from typing import override

from scipy import stats

from ..result import HypothesisTestResult
from ..two_sample_test import TwoSampleTest
from ..types import FloatArray


@dataclass(frozen=True)
class CramerVonMises(TwoSampleTest):
    """
    Two-sample Cramér-von Mises test of whether two samples come from one distribution.

    Detects any difference between the distributions from the squared distances of
    their empirical distribution functions summed over all observations, which
    makes it more powerful than the Kolmogorov-Smirnov test against many
    alternatives.
    """

    @override
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        return HypothesisTestResult.from_outcome(stats.cramervonmises_2samp(first, second))
