from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .array_types import FloatArray
from .data_validator import DataValidator
from .result import HypothesisTestResult


class OneSampleTest(ABC):
    """
    Test of a hypothesis about the distribution of one sample.

    A paired test is a one-sample test of the differences between pairs, which
    `run_paired` takes. Concrete tests are frozen dataclasses of their options. They
    implement only the computation on validated data that the test is defined for;
    this class handles validation and undefined cases.

    Attributes
    ----------
    min_samples : int
        Fewest observations for which the test is defined.
    """

    min_samples: ClassVar[int] = 2

    def run(self, sample: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on one sample.

        Parameters
        ----------
        sample : ArrayLike
            The one-dimensional observations.

        Returns
        -------
        HypothesisTestResult
            The outcome; its statistic is None when the test is undefined for the
            sample, such as when it has too few observations.

        Raises
        ------
        TypeError
            If the sample is not numeric.
        ValueError
            If the sample is not one-dimensional, is empty, or is not finite.
        """
        return self._run_validated(DataValidator.to_sample(sample, "sample"))

    def run_paired(self, first: ArrayLike, second: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on the differences between paired observations.

        Parameters
        ----------
        first : ArrayLike
            The first observation of each pair, such as before a treatment.
        second : ArrayLike
            The second observation of each pair, aligned with first.

        Returns
        -------
        HypothesisTestResult
            The outcome on first - second, as from `run`.

        Raises
        ------
        TypeError
            If an observation set is not numeric.
        ValueError
            If an observation set is invalid as for `run`, or the two are not aligned.
        """
        first_values: FloatArray = DataValidator.to_sample(first, "first")
        second_values: FloatArray = DataValidator.to_sample(second, "second")
        if first_values.size != second_values.size:
            raise ValueError(
                f"first and second must have the same length, got {first_values.size} and {second_values.size}"
            )
        return self._run_validated(first_values - second_values)

    def _run_validated(self, sample: FloatArray) -> HypothesisTestResult:
        if sample.size < self.min_samples:
            return HypothesisTestResult(statistic=None)
        # Extreme or degenerate data that validation allows ends as NaN or infinity.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if not self._is_defined(sample):
                return HypothesisTestResult(statistic=None)
            return self._compute(sample)

    def _is_defined(self, sample: FloatArray) -> bool:
        # Tests that are undefined for some data, such as constant samples, override this.
        _ = sample
        return True

    @abstractmethod
    def _compute(self, sample: FloatArray) -> HypothesisTestResult:
        """
        Compute the test on validated data.

        Parameters
        ----------
        sample : FloatArray
            The observations.

        Returns
        -------
        HypothesisTestResult
            The outcome.
        """
