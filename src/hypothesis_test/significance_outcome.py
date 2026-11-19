from typing import Protocol, SupportsFloat


class SignificanceOutcome(Protocol):
    """
    Result of a backing SciPy test.
    """

    @property
    def statistic(self) -> SupportsFloat:
        """
        Get the test's statistic.

        Returns
        -------
        SupportsFloat
            The statistic.
        """
        ...

    @property
    def pvalue(self) -> SupportsFloat:
        """
        Get the test's p-value.

        Returns
        -------
        SupportsFloat
            The p-value.
        """
        ...
