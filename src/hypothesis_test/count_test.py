from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .array_types import IntArray
from .data_validator import DataValidator
from .result import HypothesisTestResult


class CountTest(ABC):
    """
    Test of a hypothesis about categorical data summarized as frequencies.

    Concrete tests are frozen dataclasses of their options. They implement only the
    computation on validated data that the test is defined for; this class handles
    validation and undefined cases.

    Attributes
    ----------
    shape : tuple[int | None, ...]
        Shape of the frequencies the test takes, with None for a dimension of any
        length from 2.
    """

    shape: ClassVar[tuple[int | None, ...]]

    def run(self, counts: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on frequencies.

        Parameters
        ----------
        counts : ArrayLike
            The frequencies, with `shape`, as integers or as integral floats.

        Returns
        -------
        HypothesisTestResult
            The outcome; its statistic is None when the test is undefined for the
            frequencies, such as when they are all zero.

        Raises
        ------
        TypeError
            If the frequencies are not numeric.
        ValueError
            If the frequencies do not have `shape` or do not match the test's options,
            or a frequency is negative, not finite, not an integer, or too large for a
            64-bit integer.
        """
        frequencies: IntArray = DataValidator.to_counts(counts, "counts")
        if not self._has_shape(frequencies):
            expected: str = " x ".join(
                "n" if length is None else str(length) for length in self.shape
            )
            raise ValueError(
                f"{type(self).__name__} takes counts of shape {expected} with n >= 2, got {frequencies.shape}"
            )
        self._require_compatible(frequencies)
        if not np.any(frequencies):
            return HypothesisTestResult(statistic=None)
        # Extreme or degenerate data that validation allows ends as NaN or infinity.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if not self._is_defined(frequencies):
                return HypothesisTestResult(statistic=None)
            return self._compute(frequencies)

    def _has_shape(self, counts: IntArray) -> bool:
        return counts.ndim == len(self.shape) and all(
            length >= 2 if expected is None else length == expected
            for length, expected in zip(counts.shape, self.shape, strict=True)
        )

    def _require_compatible(self, counts: IntArray) -> None:
        # Tests whose options fix part of the shape, such as the number of categories,
        # override this to raise ValueError.
        _ = counts

    def _is_defined(self, counts: IntArray) -> bool:
        # Tests that are undefined for some data, such as empty categories, override this.
        _ = counts
        return True

    @abstractmethod
    def _compute(self, counts: IntArray) -> HypothesisTestResult:
        """
        Compute the test on validated data.

        Parameters
        ----------
        counts : IntArray
            The frequencies, not all zero.

        Returns
        -------
        HypothesisTestResult
            The outcome.
        """
