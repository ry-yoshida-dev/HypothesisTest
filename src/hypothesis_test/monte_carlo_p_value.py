from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_types import FloatArray, IntArray
from .options import Alternative
from .parameter_validator import ParameterValidator


class MonteCarloPValue:
    """
    p-values of observed statistics against statistics simulated under the null hypothesis.

    Use it for permutation and Monte Carlo tests. A p-value is ``(count + 1) / (simulations + 1)``,
    where the count is the number of simulated statistics at least as extreme as the observed one,
    and the added one counts the observed statistic itself. The smallest attainable p-value is
    therefore ``1 / (simulations + 1)``, which bounds what a correction for multiple comparisons
    can find.

    The two-sided p-value is twice the smaller one-sided p-value, capped at 1. Libraries such as
    PySAL/esda report the smaller one-sided p-value instead for local statistics, so p-values
    differ from theirs by that factor of two.

    Counting and converting counts are separate operations, so that simulations can arrive in
    blocks of which only the running counts are kept.
    """

    @staticmethod
    def count_extremes(observed: ArrayLike, simulated: ArrayLike) -> tuple[IntArray, IntArray]:
        """
        Count the simulated statistics at or beyond each observed statistic on either side.

        Ties count on both sides. A tolerance of ``1e-10 * max(1, |observed|)`` makes ties
        robust to the floating-point summation order of the simulation.

        Parameters
        ----------
        observed : ArrayLike
            The one-dimensional observed statistics, of shape (n,).
        simulated : ArrayLike
            The simulated statistics of each observed one, of shape (n, k).

        Returns
        -------
        tuple[IntArray, IntArray]
            Per observation, the number of simulated statistics greater than or equal to the
            observed one, and the number less than or equal to it; both of shape (n,).

        Raises
        ------
        TypeError
            If the statistics are not numeric.
        ValueError
            If the shapes do not match, an observed statistic is not finite, or a simulated
            statistic is NaN.
        """
        observed_array: FloatArray = MonteCarloPValue._as_floats("observed", observed)
        simulated_array: FloatArray = MonteCarloPValue._as_floats("simulated", simulated)
        if observed_array.ndim != 1:
            raise ValueError(f"observed must be 1-D, got shape {observed_array.shape}")
        if simulated_array.ndim != 2 or simulated_array.shape[0] != observed_array.size:
            raise ValueError(
                f"simulated must have shape ({observed_array.size}, k), got {simulated_array.shape}"
            )
        if not np.isfinite(observed_array).all():
            raise ValueError("observed must be finite")
        if np.isnan(simulated_array).any():
            raise ValueError("simulated must not contain NaN")
        tolerance: FloatArray = 1e-10 * np.maximum(1.0, np.abs(observed_array))
        greater: IntArray = np.sum(
            simulated_array >= (observed_array - tolerance)[:, None], axis=1, dtype=np.int64
        )
        less: IntArray = np.sum(
            simulated_array <= (observed_array + tolerance)[:, None], axis=1, dtype=np.int64
        )
        return greater, less

    @staticmethod
    def from_counts(
        greater_counts: ArrayLike,
        less_counts: ArrayLike,
        simulations: int,
        alternative: Alternative,
    ) -> FloatArray:
        """
        Get the p-values from counts of extreme simulated statistics.

        Parameters
        ----------
        greater_counts : ArrayLike
            The one-dimensional numbers of simulated statistics greater than or equal to the
            observed ones, possibly empty.
        less_counts : ArrayLike
            The numbers of simulated statistics less than or equal to the observed ones, of the
            same shape.
        simulations : int
            The number of simulated statistics behind each count, at least 1.
        alternative : Alternative
            The alternative hypothesis, about the value of the statistic.

        Returns
        -------
        FloatArray
            The p-values, at least ``1 / (simulations + 1)`` and at most 1.

        Raises
        ------
        TypeError
            If the simulations are not an integer, the counts are not integers, or the
            alternative is not an Alternative.
        ValueError
            If the simulations are below 1, the counts are not one-dimensional of the same
            shape, or a count lies outside [0, simulations].
        """
        # Typed as int, but callers may pass booleans or floats, which pass numeric checks.
        raw_simulations = cast(object, simulations)
        if isinstance(raw_simulations, bool | np.bool_) or not isinstance(
            raw_simulations, int | np.integer
        ):
            raise TypeError(f"simulations must be an integer, got {simulations!r}")
        if simulations < 1:
            raise ValueError(f"simulations must be at least 1, got {simulations}")
        ParameterValidator.require_members(Alternative, alternative=alternative)
        greater_array: IntArray = MonteCarloPValue._as_counts(
            "greater_counts", greater_counts, simulations
        )
        less_array: IntArray = MonteCarloPValue._as_counts("less_counts", less_counts, simulations)
        if greater_array.shape != less_array.shape:
            raise ValueError(
                "greater_counts and less_counts must have the same shape, "
                f"got {greater_array.shape} and {less_array.shape}"
            )
        denominator: float = float(simulations) + 1.0
        greater: FloatArray = (greater_array.astype(np.float64) + 1.0) / denominator
        less: FloatArray = (less_array.astype(np.float64) + 1.0) / denominator
        match alternative:
            case Alternative.GREATER:
                return greater
            case Alternative.LESS:
                return less
            case Alternative.TWO_SIDED:
                return np.minimum(1.0, 2.0 * np.minimum(greater, less))

    @staticmethod
    def from_simulations(observed: float, simulated: ArrayLike, alternative: Alternative) -> float:
        """
        Get the p-value of one observed statistic from all its simulated statistics.

        Parameters
        ----------
        observed : float
            The observed statistic.
        simulated : ArrayLike
            The one-dimensional simulated statistics, at least one.
        alternative : Alternative
            The alternative hypothesis, about the value of the statistic.

        Returns
        -------
        float
            The p-value, at least ``1 / (len(simulated) + 1)`` and at most 1.

        Raises
        ------
        TypeError
            If a statistic is not numeric or the alternative is not an Alternative.
        ValueError
            If the observed statistic is not finite, the simulated statistics are not
            one-dimensional or are empty, or a simulated statistic is NaN.
        """
        ParameterValidator.require_finite_numbers(observed=observed)
        simulated_array: FloatArray = MonteCarloPValue._as_floats("simulated", simulated)
        if simulated_array.ndim != 1:
            raise ValueError(f"simulated must be 1-D, got shape {simulated_array.shape}")
        if simulated_array.size == 0:
            raise ValueError("simulated must not be empty")
        greater, less = MonteCarloPValue.count_extremes(
            np.array([observed], dtype=np.float64), simulated_array[None, :]
        )
        return float(
            MonteCarloPValue.from_counts(greater, less, simulated_array.size, alternative)[0]
        )

    @staticmethod
    def _as_floats(name: str, values: ArrayLike) -> FloatArray:
        array: NDArray[np.generic] = np.asarray(values)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
        # Simulations arrive in large blocks, so float input is used without a copy.
        return array.astype(np.float64, copy=False)

    @staticmethod
    def _as_counts(name: str, counts: ArrayLike, simulations: int) -> IntArray:
        array: NDArray[np.generic] = np.asarray(counts)
        # An empty list becomes a float array, which holds no non-integer count.
        if array.dtype.kind not in "iu" and array.size > 0:
            raise TypeError(f"{name} must be integers, got dtype {array.dtype}")
        if array.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
        values: IntArray = array.astype(np.int64)
        if not np.all((values >= 0) & (values <= simulations)):
            raise ValueError(f"{name} must be in [0, {simulations}]")
        return values
