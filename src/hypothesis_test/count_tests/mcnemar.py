from dataclasses import dataclass
from typing import override

from scipy import stats

from ..count_test import CountTest
from ..options import McNemarVariant
from ..parameter_validator import ParameterValidator
from ..result import HypothesisTestResult
from ..types import IntArray


@dataclass(frozen=True)
class McNemar(CountTest):
    """
    McNemar's test of whether paired binary outcomes change in either direction equally often.

    Takes the 2 x 2 table of the outcomes of each pair, such as before and after a
    treatment, and uses only the discordant pairs off its diagonal. The statistic is
    the smaller discordant count for the exact test, and the chi-square statistic
    otherwise. Undefined without discordant pairs.

    Attributes
    ----------
    variant : McNemarVariant
        Reference distribution of the test.
    """

    variant: McNemarVariant = McNemarVariant.EXACT

    shape = (2, 2)

    def __post_init__(self) -> None:
        ParameterValidator.require_members(McNemarVariant, variant=self.variant)

    @override
    def _is_defined(self, counts: IntArray) -> bool:
        return bool(counts[0, 1] + counts[1, 0] > 0)

    @override
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        # SciPy has no McNemar test, so it follows the published formulas.
        first_discordant: int = int(counts[0, 1])
        second_discordant: int = int(counts[1, 0])
        num_discordant: int = first_discordant + second_discordant
        match self.variant:
            case McNemarVariant.EXACT:
                smaller: int = min(first_discordant, second_discordant)
                return HypothesisTestResult(
                    statistic=float(smaller),
                    p_value=float(stats.binomtest(smaller, num_discordant).pvalue),
                )
            case McNemarVariant.ASYMPTOTIC:
                return self._chi_square(
                    (first_discordant - second_discordant) ** 2 / num_discordant
                )
            case McNemarVariant.CORRECTED:
                corrected_gap: int = max(abs(first_discordant - second_discordant) - 1, 0)
                return self._chi_square(corrected_gap**2 / num_discordant)

    @staticmethod
    def _chi_square(statistic: float) -> HypothesisTestResult:
        return HypothesisTestResult(
            statistic=statistic,
            p_value=float(stats.chi2.sf(statistic, 1)),
            degrees_of_freedom=(1,),
        )
