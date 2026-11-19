from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data_validator import DataValidator
from .result import HypothesisTestResult
from .types import FloatArray, IntArray


class MultiSampleTest(ABC):
    """
    Test of a hypothesis comparing the distributions of several samples.

    Concrete tests are frozen dataclasses of their options. They implement only the
    computation on validated data that the test is defined for; this class handles
    validation and undefined cases.

    Attributes
    ----------
    min_groups : int
        Fewest samples the test compares.
    min_samples : int
        Fewest observations in each sample for which the test is defined.
    is_repeated_measures : bool
        Whether the samples are measurements of the same subjects, so that the i-th
        observations of all samples form one block.
    """

    min_groups: ClassVar[int] = 2
    min_samples: ClassVar[int] = 2
    is_repeated_measures: ClassVar[bool] = False

    def run(self, *samples: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on several samples.

        Parameters
        ----------
        *samples : ArrayLike
            The one-dimensional observations of each sample.

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
            If there are fewer samples than `min_groups`, a sample is not
            one-dimensional, is empty, or is not finite, or repeated measures are not
            aligned.
        """
        if len(samples) < self.min_groups:
            raise ValueError(
                f"{type(self).__name__} compares at least {self.min_groups} samples, got {len(samples)}"
            )
        validated: list[FloatArray] = [
            DataValidator.to_sample(sample, f"sample {index}")
            for index, sample in enumerate(samples)
        ]
        sizes: list[int] = [sample.size for sample in validated]
        if self.is_repeated_measures and len(set(sizes)) > 1:
            raise ValueError(f"repeated measures must have the same length, got {sizes}")
        if min(sizes) < self.min_samples:
            return HypothesisTestResult(statistic=None)
        # Extreme or degenerate data that validation allows ends as NaN or infinity.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if not self._is_defined(validated):
                return HypothesisTestResult(statistic=None)
            return self._compute(validated)

    def run_grouped(self, values: ArrayLike, groups: ArrayLike) -> HypothesisTestResult:
        """
        Run the test on observations labeled by their sample, as in a long-format table.

        Parameters
        ----------
        values : ArrayLike
            The one-dimensional observations of all samples.
        groups : ArrayLike
            The sample label of each observation, aligned with values. Samples are
            ordered by sorted label, and observations keep their order within a
            sample, which aligns the blocks of repeated measures.

        Returns
        -------
        HypothesisTestResult
            The outcome, as from `run`.

        Raises
        ------
        TypeError
            If the observations are not numeric, or the labels cannot be sorted.
        ValueError
            If the observations are invalid as for `run`, or the labels are not
            one-dimensional or not aligned with the observations.
        """
        observations: FloatArray = DataValidator.to_sample(values, "values")
        labels: NDArray[np.generic] = np.asarray(groups)
        if labels.shape != observations.shape:
            raise ValueError(
                f"groups must have the shape of values {observations.shape}, got {labels.shape}"
            )
        codes: IntArray = np.unique(labels, return_inverse=True)[1].astype(np.int64)
        order: IntArray = np.argsort(codes, kind="stable")
        boundaries: IntArray = np.cumsum(np.bincount(codes))[:-1]
        return self.run(*np.split(observations[order], boundaries))

    def _is_defined(self, samples: list[FloatArray]) -> bool:
        # Tests that are undefined for some data, such as constant samples, override this.
        _ = samples
        return True

    @abstractmethod
    def _compute(self, samples: list[FloatArray]) -> HypothesisTestResult:
        """
        Compute the test on validated data.

        Parameters
        ----------
        samples : list[FloatArray]
            The samples.

        Returns
        -------
        HypothesisTestResult
            The outcome.
        """
