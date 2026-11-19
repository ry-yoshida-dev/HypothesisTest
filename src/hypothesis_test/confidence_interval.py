from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    Confidence interval of the parameter a test is about, such as a mean or a proportion.

    One-sided alternatives give an interval with one infinite bound.

    Attributes
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.
    level : float
        Confidence level in (0, 1).
    """

    low: float
    high: float
    level: float
