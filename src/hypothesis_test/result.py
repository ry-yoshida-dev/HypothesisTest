from __future__ import annotations

import math
from dataclasses import dataclass

from .confidence_interval import ConfidenceInterval
from .parameter_validator import ParameterValidator
from .significance_outcome import SignificanceOutcome

# Backing tests can round a p-value of 1 up past it by a few units in the last place.
_P_VALUE_ROUNDING_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class HypothesisTestResult:
    """
    Outcome of a hypothesis test.

    NaN from a backing library is stored as None, so undefined values cannot silently
    propagate through later arithmetic. A p-value is at most 1. An undefined statistic
    has neither a p-value, degrees of freedom, nor a confidence interval.

    Attributes
    ----------
    statistic : float | None
        The test statistic; None when the test is undefined for the data, such as for
        constant samples.
    p_value : float | None
        The p-value; None when it is undefined although the statistic is not, such as
        when every observation is tied.
    degrees_of_freedom : tuple[float, ...]
        The degrees of freedom of the reference distribution: one for t and
        chi-square distributions, the numerator and the denominator for F
        distributions, and none for other distributions.
    confidence_interval : ConfidenceInterval | None
        The confidence interval of the tested parameter; None when the test does not
        provide one.

    Raises
    ------
    ValueError
        If the p-value is outside [0, 1].
    """

    statistic: float | None
    p_value: float | None = None
    degrees_of_freedom: tuple[float, ...] = ()
    confidence_interval: ConfidenceInterval | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "degrees_of_freedom", tuple(float(value) for value in self.degrees_of_freedom)
        )
        if self.statistic is not None and math.isnan(self.statistic):
            object.__setattr__(self, "statistic", None)
        if self.statistic is None:
            object.__setattr__(self, "p_value", None)
            object.__setattr__(self, "degrees_of_freedom", ())
            object.__setattr__(self, "confidence_interval", None)
        elif self.p_value is not None and math.isnan(self.p_value):
            object.__setattr__(self, "p_value", None)
        elif self.p_value is not None:
            if not 0 <= self.p_value <= 1 + _P_VALUE_ROUNDING_TOLERANCE:
                raise ValueError(f"p_value must be in [0, 1], got {self.p_value!r}")
            object.__setattr__(self, "p_value", min(self.p_value, 1.0))

    @classmethod
    def from_outcome(
        cls,
        outcome: SignificanceOutcome,
        *,
        degrees_of_freedom: tuple[float, ...] = (),
    ) -> HypothesisTestResult:
        """
        Build a result from a backing SciPy test.

        Parameters
        ----------
        outcome : SignificanceOutcome
            The test's statistic and p-value.
        degrees_of_freedom : tuple[float, ...]
            The degrees of freedom of the reference distribution.

        Returns
        -------
        HypothesisTestResult
            The result.
        """
        return cls(
            statistic=float(outcome.statistic),
            p_value=float(outcome.pvalue),
            degrees_of_freedom=degrees_of_freedom,
        )

    def is_significant(self, significance_level: float = 0.05) -> bool:
        """
        Check whether the test rejects the null hypothesis at a significance level.

        Parameters
        ----------
        significance_level : float
            The significance level in (0, 1).

        Returns
        -------
        bool
            True if the p-value is at most the significance level; False when the
            p-value is undefined.

        Raises
        ------
        TypeError
            If the significance level is not a real number.
        ValueError
            If the significance level is outside (0, 1).
        """
        ParameterValidator.require_open_unit_interval(significance_level=significance_level)
        return self.p_value is not None and self.p_value <= significance_level
