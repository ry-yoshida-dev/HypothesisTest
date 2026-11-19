from enum import Enum
from typing import Literal


class DivergenceStatistic(Enum):
    """
    Statistic that measures how far observed frequencies are from expected ones.

    Both follow the same chi-square distribution asymptotically.

    Attributes
    ----------
    PEARSON
        Pearson's chi-square statistic.
    LOG_LIKELIHOOD
        The likelihood-ratio statistic G, which adds up over nested models.
    """

    PEARSON = "pearson"
    LOG_LIKELIHOOD = "log-likelihood"

    @property
    def option_name(self) -> Literal["pearson", "log-likelihood"]:
        """
        Get the option name that the backing tests accept.

        Returns
        -------
        Literal["pearson", "log-likelihood"]
            The option name.
        """
        match self:
            case DivergenceStatistic.PEARSON:
                return "pearson"
            case DivergenceStatistic.LOG_LIKELIHOOD:
                return "log-likelihood"
