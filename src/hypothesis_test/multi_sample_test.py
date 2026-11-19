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

    def run_grouped(
        self, values: ArrayLike, groups: ArrayLike, subjects: ArrayLike | None = None
    ) -> HypothesisTestResult:
        """
        Run the test on observations labeled by their sample, as in a long-format table.

        Parameters
        ----------
        values : ArrayLike
            The one-dimensional observations of all samples.
        groups : ArrayLike
            The sample label of each observation, aligned with values. Samples are
            ordered by sorted label.
        subjects : ArrayLike | None
            The subject label of each observation, aligned with values, which pairs the
            measurements of each subject across samples whatever the row order. Required
            when `is_repeated_measures` is True, and None otherwise.

        Returns
        -------
        HypothesisTestResult
            The outcome, as from `run`.

        Raises
        ------
        TypeError
            If the observations are not numeric, or the labels cannot be sorted.
        ValueError
            If the observations are invalid as for `run`; the labels are not aligned
            with the observations; subjects are missing for repeated measures or given
            for independent samples; or a subject does not have exactly one
            observation in each sample.
        """
        observations: FloatArray = DataValidator.to_sample(values, "values")
        group_codes: IntArray = self._label_codes(groups, "groups", observations.shape)
        name: str = type(self).__name__
        order: IntArray
        if not self.is_repeated_measures:
            if subjects is not None:
                raise ValueError(f"{name} compares independent samples, so it takes no subjects")
            order = np.argsort(group_codes, kind="stable")
        else:
            if subjects is None:
                raise ValueError(f"{name} compares repeated measures, so it requires subjects")
            subject_codes: IntArray = self._label_codes(subjects, "subjects", observations.shape)
            num_groups: int = int(group_codes.max()) + 1
            num_subjects: int = int(subject_codes.max()) + 1
            cell_sizes: IntArray = np.bincount(
                group_codes * num_subjects + subject_codes, minlength=num_groups * num_subjects
            )
            if np.any(cell_sizes != 1):
                raise ValueError("each subject must have exactly one observation in each group")
            # Sorting by subject within each group aligns the i-th observations.
            order = np.lexsort((subject_codes, group_codes))
        boundaries: IntArray = np.cumsum(np.bincount(group_codes))[:-1]
        return self.run(*np.split(observations[order], boundaries))

    @staticmethod
    def _label_codes(labels: ArrayLike, name: str, shape: tuple[int, ...]) -> IntArray:
        array: NDArray[np.generic] = np.asarray(labels)
        if array.shape != shape:
            raise ValueError(f"{name} must have the shape of values {shape}, got {array.shape}")
        codes: IntArray = np.unique(array, return_inverse=True)[1].astype(np.int64)
        return codes

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
