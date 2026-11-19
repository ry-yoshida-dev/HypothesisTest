import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import FloatArray, IntArray


class DataValidator:
    """
    Conversion of the data passed to a test into validated arrays.
    """

    @staticmethod
    def to_sample(values: ArrayLike, name: str) -> FloatArray:
        """
        Convert observations into a one-dimensional float array.

        Parameters
        ----------
        values : ArrayLike
            The observations.
        name : str
            The name of the data in error messages.

        Returns
        -------
        FloatArray
            The observations.

        Raises
        ------
        TypeError
            If the observations are not numeric; booleans are rejected.
        ValueError
            If the observations are not one-dimensional, are empty, or are not finite.
        """
        array: NDArray[np.generic] = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        if array.size == 0:
            raise ValueError(f"{name} must not be empty")
        sample: FloatArray = array.astype(np.float64)
        if not np.all(np.isfinite(sample)):
            raise ValueError(f"{name} must be finite")
        return sample

    @staticmethod
    def to_counts(values: ArrayLike, name: str) -> IntArray:
        """
        Convert frequencies, such as a contingency table, into an integer array.

        Parameters
        ----------
        values : ArrayLike
            The frequencies, as integers or as integral floats.
        name : str
            The name of the data in error messages.

        Returns
        -------
        IntArray
            The frequencies, with the shape of the input.

        Raises
        ------
        TypeError
            If the frequencies are not numeric; booleans are rejected.
        ValueError
            If a frequency is negative, not finite, or not an integer.
        """
        array: NDArray[np.generic] = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
        counts: FloatArray = array.astype(np.float64)
        if not np.all(np.isfinite(counts)):
            raise ValueError(f"{name} must be finite")
        if np.any(counts < 0):
            raise ValueError(f"{name} must not be negative")
        if np.any(counts != np.round(counts)):
            raise ValueError(f"{name} must contain integer counts")
        return counts.astype(np.int64)
