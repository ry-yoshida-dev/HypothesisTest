from dataclasses import dataclass
from typing import override

from scipy import stats

from ..array_types import IntArray
from ..confidence_interval import ConfidenceInterval
from ..count_test import CountTest
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult


@dataclass(frozen=True)
class Binomial(CountTest):
    """
    Exact binomial test of whether the probability of success equals a value.

    Takes the numbers of successes and failures. The statistic is the observed
    proportion of successes, and the confidence interval is the Clopper-Pearson
    interval of the probability of success.

    Attributes
    ----------
    expected_proportion : float
        Probability of success under the null hypothesis, in (0, 1).
    alternative : Alternative
        Alternative hypothesis about the probability of success.
    confidence_level : float
        Level of the confidence interval of the probability of success, in (0, 1).
    """

    expected_proportion: float = 0.5
    alternative: Alternative = Alternative.TWO_SIDED
    confidence_level: float = 0.95

    shape = (2,)

    def __post_init__(self) -> None:
        ParameterValidator.require_open_unit_interval(
            expected_proportion=self.expected_proportion, confidence_level=self.confidence_level
        )
        ParameterValidator.require_members(Alternative, alternative=self.alternative)

    @override
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        num_successes: int = int(counts[0])
        outcome = stats.binomtest(
            num_successes,
            num_successes + int(counts[1]),
            p=self.expected_proportion,
            alternative=self.alternative.option_name,
        )
        interval = outcome.proportion_ci(confidence_level=self.confidence_level)
        return HypothesisTestResult(
            statistic=float(outcome.statistic),
            p_value=float(outcome.pvalue),
            confidence_interval=ConfidenceInterval(
                low=float(interval.low), high=float(interval.high), level=self.confidence_level
            ),
        )
