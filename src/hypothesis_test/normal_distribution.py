import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .array_types import FloatArray
from .options import Alternative
from .parameter_validator import ParameterValidator


class NormalDistribution:
    """
    The standard normal distribution as the reference of a z statistic.

    The distribution has no parameters, so its p-values are static methods.
    """

    @staticmethod
    def p_value(statistic: float, alternative: Alternative) -> float:
        """
        Get the p-value of a z statistic.

        Parameters
        ----------
        statistic : float
            The z statistic.
        alternative : Alternative
            The alternative hypothesis.

        Returns
        -------
        float
            The p-value; NaN if the statistic is NaN.

        Raises
        ------
        TypeError
            If the alternative is not an Alternative.
        """
        ParameterValidator.require_members(Alternative, alternative=alternative)
        match alternative:
            case Alternative.TWO_SIDED:
                return 2 * float(stats.norm.sf(abs(statistic)))
            case Alternative.GREATER:
                return float(stats.norm.sf(statistic))
            case Alternative.LESS:
                return float(stats.norm.cdf(statistic))

    @staticmethod
    def p_values(statistics: ArrayLike, alternative: Alternative) -> FloatArray:
        """
        Get the p-values of z statistics.

        Parameters
        ----------
        statistics : ArrayLike
            The one-dimensional z statistics.
        alternative : Alternative
            The alternative hypothesis.

        Returns
        -------
        FloatArray
            The p-values in the order of the input; NaN where the statistic is NaN.

        Raises
        ------
        TypeError
            If the statistics are not numeric or the alternative is not an Alternative.
        ValueError
            If the statistics are not one-dimensional.
        """
        ParameterValidator.require_members(Alternative, alternative=alternative)
        array: NDArray[np.generic] = np.asarray(statistics)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"statistics must be numeric, got dtype {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"statistics must be 1-D, got shape {array.shape}")
        values: FloatArray = array.astype(np.float64)
        match alternative:
            case Alternative.TWO_SIDED:
                return np.asarray(2 * stats.norm.sf(np.abs(values)), dtype=np.float64)
            case Alternative.GREATER:
                return np.asarray(stats.norm.sf(values), dtype=np.float64)
            case Alternative.LESS:
                return np.asarray(stats.norm.cdf(values), dtype=np.float64)
