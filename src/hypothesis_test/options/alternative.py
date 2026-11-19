from enum import Enum
from typing import Literal


class Alternative(Enum):
    """
    Alternative hypothesis of a test, relative to the value under the null hypothesis.

    For two samples, GREATER means that the first is greater than the second.

    Attributes
    ----------
    TWO_SIDED
        The tested quantity differs from its null value.
    GREATER
        The tested quantity is greater than its null value.
    LESS
        The tested quantity is less than its null value.
    """

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"

    @property
    def option_name(self) -> Literal["two-sided", "greater", "less"]:
        """
        Get the option name that the backing tests accept.

        Returns
        -------
        Literal["two-sided", "greater", "less"]
            The option name.
        """
        match self:
            case Alternative.TWO_SIDED:
                return "two-sided"
            case Alternative.GREATER:
                return "greater"
            case Alternative.LESS:
                return "less"
