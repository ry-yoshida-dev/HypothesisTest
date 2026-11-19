from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .data_validator import DataValidator
from .result import HypothesisTestResult
from .types import FloatArray


class TwoSampleTest(ABC):
    """
    Test of a hypothesis comparing the distributions of two independent samples.

    Concrete tests are frozen dataclasses of their options. They implement only the
    computation on validated data that the test is defined for; this class handles
    validation and undefined cases.

    Attributes
    ----------
    min_samples : int
        Fewest observations in each sample for which the test is defined.
    """

    min_samples: ClassVar[int] = 2

    def run(self, first: ArrayLike, second: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on two independent samples.

        Parameters
        ----------
        first : ArrayLike
            The one-dimensional observations of the first sample.
        second : ArrayLike
            The one-dimensional observations of the second sample, of any length.

        Returns
        -------
        HypothesisTestResult
            The outcome; its statistic is None when the test is undefined for the
            samples, such as when one has too few observations.

        Raises
        ------
        TypeError
            If a sample is not numeric.
        ValueError
            If a sample is not one-dimensional, is empty, or is not finite.
        """
        first_values: FloatArray = DataValidator.to_sample(first, "first")
        second_values: FloatArray = DataValidator.to_sample(second, "second")
        if min(first_values.size, second_values.size) < self.min_samples:
            return HypothesisTestResult(statistic=None)
        # Extreme or degenerate data that validation allows ends as NaN or infinity.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if not self._is_defined(first_values, second_values):
                return HypothesisTestResult(statistic=None)
            return self._compute(first_values, second_values)

    def _is_defined(self, first: FloatArray, second: FloatArray) -> bool:
        # Tests that are undefined for some data, such as constant samples, override this.
        _ = first, second
        return True

    @abstractmethod
    def _compute(self, first: FloatArray, second: FloatArray) -> HypothesisTestResult:
        """
        Compute the test on validated data.

        Parameters
        ----------
        first : FloatArray
            The first sample.
        second : FloatArray
            The second sample.

        Returns
        -------
        HypothesisTestResult
            The outcome.
        """
