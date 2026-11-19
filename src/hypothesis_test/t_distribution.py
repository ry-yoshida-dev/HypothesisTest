import math
from dataclasses import dataclass

from scipy import stats

from .confidence_interval import ConfidenceInterval
from .options import Alternative


@dataclass(frozen=True)
class TDistribution:
    """
    Student's t distribution as the reference of a t statistic.

    Attributes
    ----------
    degrees_of_freedom : float
        The degrees of freedom, positive and possibly fractional.
    """

    degrees_of_freedom: float

    def p_value(self, statistic: float, alternative: Alternative) -> float:
        """
        Get the p-value of a t statistic.

        Parameters
        ----------
        statistic : float
            The t statistic.
        alternative : Alternative
            The alternative hypothesis.

        Returns
        -------
        float
            The p-value.
        """
        match alternative:
            case Alternative.TWO_SIDED:
                return 2 * float(stats.t.sf(abs(statistic), self.degrees_of_freedom))
            case Alternative.GREATER:
                return float(stats.t.sf(statistic, self.degrees_of_freedom))
            case Alternative.LESS:
                return float(stats.t.cdf(statistic, self.degrees_of_freedom))

    def confidence_interval(
        self,
        estimate: float,
        standard_error: float,
        alternative: Alternative,
        level: float,
    ) -> ConfidenceInterval:
        """
        Get the confidence interval of an estimate that the t statistic standardizes.

        Parameters
        ----------
        estimate : float
            The estimate, such as a sample mean.
        standard_error : float
            The standard error of the estimate.
        alternative : Alternative
            The alternative hypothesis, which makes the interval one-sided unless it is
            two-sided.
        level : float
            The confidence level in (0, 1).

        Returns
        -------
        ConfidenceInterval
            The interval.
        """
        match alternative:
            case Alternative.TWO_SIDED:
                margin: float = self._quantile((1 + level) / 2) * standard_error
                return ConfidenceInterval(
                    low=estimate - margin, high=estimate + margin, level=level
                )
            case Alternative.GREATER:
                return ConfidenceInterval(
                    low=estimate - self._quantile(level) * standard_error,
                    high=math.inf,
                    level=level,
                )
            case Alternative.LESS:
                return ConfidenceInterval(
                    low=-math.inf,
                    high=estimate + self._quantile(level) * standard_error,
                    level=level,
                )

    def _quantile(self, probability: float) -> float:
        return float(stats.t.ppf(probability, self.degrees_of_freedom))
