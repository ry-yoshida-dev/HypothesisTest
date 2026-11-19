from enum import Enum
from typing import Literal

import numpy as np

from ..types import FloatArray


class Center(Enum):
    """
    Center from which a test of equal variances measures deviations.

    Attributes
    ----------
    MEAN
        The group mean, the original form of Levene's test, most powerful for
        normal data.
    MEDIAN
        The group median, robust to skewed and heavy-tailed data; Levene's test
        with it is the Brown-Forsythe test.
    """

    MEAN = "mean"
    MEDIAN = "median"

    @property
    def option_name(self) -> Literal["mean", "median"]:
        """
        Get the option name that the backing tests accept.

        Returns
        -------
        Literal["mean", "median"]
            The option name.
        """
        match self:
            case Center.MEAN:
                return "mean"
            case Center.MEDIAN:
                return "median"

    def locate(self, sample: FloatArray) -> float:
        """
        Get the center of a sample.

        Parameters
        ----------
        sample : FloatArray
            The observations.

        Returns
        -------
        float
            The mean or the median of the observations.
        """
        match self:
            case Center.MEAN:
                return float(np.mean(sample))
            case Center.MEDIAN:
                return float(np.median(sample))
