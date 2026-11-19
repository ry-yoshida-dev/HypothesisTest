from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats
from scipy.stats import contingency

from ..confidence_interval import ConfidenceInterval
from ..count_test import CountTest
from ..options import Alternative
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import IntArray


@dataclass(frozen=True)
class FisherExact(CountTest):
    """
    Fisher's exact test of whether the odds ratio of a 2 x 2 contingency table is one.

    Exact for any sample size, conditioning on the row and column totals. The
    statistic is the conditional maximum likelihood estimate of the odds ratio, the
    estimate whose confidence interval matches the test. Undefined when a row or a
    column has no observations.

    Attributes
    ----------
    alternative : Alternative
        Alternative hypothesis about the odds ratio relative to one.
    confidence_level : float
        Level of the confidence interval of the odds ratio, in (0, 1).
    """

    alternative: Alternative = Alternative.TWO_SIDED
    confidence_level: float = 0.95

    shape = (2, 2)

    def __post_init__(self) -> None:
        ParameterValidator.require_members(Alternative, alternative=self.alternative)
        ParameterValidator.require_open_unit_interval(confidence_level=self.confidence_level)

    @override
    def _is_defined(self, counts: IntArray) -> bool:
        return bool(np.all(np.sum(counts, axis=0) > 0) and np.all(np.sum(counts, axis=1) > 0))

    @override
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        odds_ratio = contingency.odds_ratio(counts, kind="conditional")
        interval = odds_ratio.confidence_interval(
            confidence_level=self.confidence_level, alternative=self.alternative.option_name
        )
        outcome = stats.fisher_exact(counts, alternative=self.alternative.option_name)
        return HypothesisTestResult(
            statistic=float(odds_ratio.statistic),
            p_value=float(outcome.pvalue),
            confidence_interval=ConfidenceInterval(
                low=float(interval.low), high=float(interval.high), level=self.confidence_level
            ),
        )
