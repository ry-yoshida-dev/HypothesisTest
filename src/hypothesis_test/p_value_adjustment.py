from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .array_types import FloatArray, IntArray


class PValueAdjustment(Enum):
    """
    Adjustment of the p-values of several tests for multiple comparisons.

    Rejecting each hypothesis whose adjusted p-value is at most a level controls the
    family-wise error rate or the false discovery rate at that level.

    Attributes
    ----------
    BONFERRONI
        Controls the family-wise error rate under any dependence.
    HOLM
        Controls the family-wise error rate under any dependence, and rejects at
        least as many hypotheses as BONFERRONI.
    BENJAMINI_HOCHBERG
        Controls the false discovery rate for independent or positively dependent
        tests.
    BENJAMINI_YEKUTIELI
        Controls the false discovery rate under any dependence.
    """

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini-hochberg"
    BENJAMINI_YEKUTIELI = "benjamini-yekutieli"

    def adjust(self, p_values: ArrayLike) -> FloatArray:
        """
        Adjust p-values.

        Parameters
        ----------
        p_values : ArrayLike
            The one-dimensional p-values of the tests.

        Returns
        -------
        FloatArray
            The adjusted p-values, in the order of the input.

        Raises
        ------
        TypeError
            If the p-values are not numeric.
        ValueError
            If the p-values are not one-dimensional or lie outside [0, 1].
        """
        array: NDArray[np.generic] = np.asarray(p_values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"p_values must be numeric, got dtype {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"p_values must be 1-D, got shape {array.shape}")
        values: FloatArray = array.astype(np.float64)
        if not np.all((values >= 0) & (values <= 1)):
            raise ValueError("p_values must be in [0, 1]")
        if values.size == 0:
            return values
        match self:
            case PValueAdjustment.BONFERRONI:
                return np.minimum(values * values.size, 1.0)
            case PValueAdjustment.HOLM:
                return self._holm(values)
            case PValueAdjustment.BENJAMINI_HOCHBERG:
                return np.asarray(
                    stats.false_discovery_control(values, method="bh"), dtype=np.float64
                )
            case PValueAdjustment.BENJAMINI_YEKUTIELI:
                return np.asarray(
                    stats.false_discovery_control(values, method="by"), dtype=np.float64
                )

    @staticmethod
    def _holm(values: FloatArray) -> FloatArray:
        # The k-th smallest p-value is scaled by the number of hypotheses not yet
        # rejected, and the running maximum keeps the adjusted values monotonic.
        order: IntArray = np.argsort(values, kind="stable")
        multipliers: FloatArray = np.arange(values.size, 0, -1, dtype=np.float64)
        scaled: FloatArray = np.minimum(np.maximum.accumulate(values[order] * multipliers), 1.0)
        adjusted: FloatArray = np.empty_like(values)
        adjusted[order] = scaled
        return adjusted
