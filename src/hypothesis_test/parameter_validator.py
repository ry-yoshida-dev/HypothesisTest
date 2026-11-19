from __future__ import annotations

import math
from enum import Enum
from numbers import Real

import numpy as np


class ParameterValidator:
    """
    Runtime checks of test options that their types do not guarantee.

    NaN compares false to everything, booleans pass numeric range checks, any value
    has a truth value, and other values reach enumeration dispatch, so all would
    otherwise fail later with an unclear error or not at all.
    """

    @staticmethod
    def require_booleans(**values: object) -> None:
        """
        Check that values are booleans.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a boolean.
        """
        for name, value in values.items():
            if not isinstance(value, bool | np.bool_):
                raise TypeError(f"{name} must be a boolean, got {value!r}")

    @staticmethod
    def require_members(enum_type: type[Enum], **values: object) -> None:
        """
        Check that values are members of an enumeration.

        Parameters
        ----------
        enum_type : type[Enum]
            The enumeration.
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a member.
        """
        for name, value in values.items():
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}, got {value!r}")

    @staticmethod
    def require_finite_numbers(**values: object) -> None:
        """
        Check that values are finite real numbers.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a real number; booleans are rejected.
        ValueError
            If a value is NaN or infinite.
        """
        for name, value in values.items():
            if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

    @staticmethod
    def require_open_unit_interval(**values: object) -> None:
        """
        Check that values are real numbers in (0, 1), such as levels and proportions.

        Parameters
        ----------
        **values : object
            The values by parameter name.

        Raises
        ------
        TypeError
            If a value is not a real number; booleans are rejected.
        ValueError
            If a value is outside (0, 1).
        """
        for name, value in values.items():
            if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number, got {value!r}")
            if not 0 < float(value) < 1:
                raise ValueError(f"{name} must be in (0, 1), got {value!r}")
