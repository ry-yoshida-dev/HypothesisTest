from dataclasses import dataclass
from typing import override

import numpy as np
from scipy import stats

from ..count_test import CountTest
from ..options import DivergenceStatistic
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import IntArray


@dataclass(frozen=True)
class ChiSquareIndependence(CountTest):
    """
    Chi-square test of whether the row and column variables of a contingency table are independent.

    The chi-square approximation needs expected frequencies of about five or more;
    use `FisherExact` for small 2 x 2 tables. Undefined when a row or a column has
    no observations.

    Attributes
    ----------
    statistic : DivergenceStatistic
        Statistic that compares observed and expected frequencies.
    is_corrected : bool
        Whether to apply Yates' continuity correction, which only affects 2 x 2
        tables.
    """

    statistic: DivergenceStatistic = DivergenceStatistic.PEARSON
    is_corrected: bool = True

    shape = (None, None)

    def __post_init__(self) -> None:
        ParameterValidator.require_members(DivergenceStatistic, statistic=self.statistic)
        ParameterValidator.require_booleans(is_corrected=self.is_corrected)

    @override
    def _is_defined(self, counts: IntArray) -> bool:
        return bool(np.all(np.sum(counts, axis=0) > 0) and np.all(np.sum(counts, axis=1) > 0))

    @override
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        outcome = stats.chi2_contingency(
            counts, correction=self.is_corrected, lambda_=self.statistic.option_name
        )
        return HypothesisTestResult.from_outcome(outcome, degrees_of_freedom=(float(outcome.dof),))
