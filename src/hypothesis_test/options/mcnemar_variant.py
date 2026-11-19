from enum import Enum


class McNemarVariant(Enum):
    """
    Reference distribution of McNemar's test.

    Attributes
    ----------
    EXACT
        The binomial distribution of the discordant pairs, valid for any count.
    ASYMPTOTIC
        The chi-square approximation, for many discordant pairs.
    CORRECTED
        The chi-square approximation with Edwards' continuity correction.
    """

    EXACT = "exact"
    ASYMPTOTIC = "asymptotic"
    CORRECTED = "corrected"
