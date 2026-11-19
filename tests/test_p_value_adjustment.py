import numpy as np
import pytest

from hypothesis_test import PValueAdjustment

P_VALUES: list[float] = [0.01, 0.04, 0.03, 0.005]


def test_p_value_adjustment_bonferroni_multiplies_by_the_number_of_tests() -> None:
    np.testing.assert_allclose(
        PValueAdjustment.BONFERRONI.adjust([0.01, 0.3, 0.2]), [0.03, 0.9, 0.6]
    )
    np.testing.assert_allclose(PValueAdjustment.BONFERRONI.adjust([0.5, 0.6]), [1.0, 1.0])


def test_p_value_adjustment_holm_matches_the_step_down_procedure() -> None:
    # Sorted: 0.005 * 4, 0.01 * 3, 0.03 * 2, 0.04 * 1, then the running maximum.
    np.testing.assert_allclose(PValueAdjustment.HOLM.adjust(P_VALUES), [0.03, 0.06, 0.06, 0.02])


def test_p_value_adjustment_holm_is_at_most_bonferroni() -> None:
    p_values = np.random.default_rng(0).uniform(size=50)
    assert np.all(
        PValueAdjustment.HOLM.adjust(p_values) <= PValueAdjustment.BONFERRONI.adjust(p_values)
    )


def test_p_value_adjustment_benjamini_hochberg_matches_the_step_up_procedure() -> None:
    # Sorted: 0.005 * 4 / 1, 0.01 * 4 / 2, 0.03 * 4 / 3, 0.04 * 4 / 4, then the
    # running minimum from the largest.
    np.testing.assert_allclose(
        PValueAdjustment.BENJAMINI_HOCHBERG.adjust(P_VALUES), [0.02, 0.04, 0.04, 0.02]
    )


def test_p_value_adjustment_benjamini_yekutieli_scales_benjamini_hochberg() -> None:
    harmonic: float = sum(1 / rank for rank in range(1, len(P_VALUES) + 1))
    np.testing.assert_allclose(
        PValueAdjustment.BENJAMINI_YEKUTIELI.adjust(P_VALUES),
        np.minimum(PValueAdjustment.BENJAMINI_HOCHBERG.adjust(P_VALUES) * harmonic, 1.0),
    )


@pytest.mark.parametrize("adjustment", list(PValueAdjustment))
def test_p_value_adjustment_keeps_one_test_and_empty_input(adjustment: PValueAdjustment) -> None:
    np.testing.assert_allclose(adjustment.adjust([0.2]), [0.2])
    assert adjustment.adjust([]).shape == (0,)


@pytest.mark.parametrize("adjustment", list(PValueAdjustment))
def test_p_value_adjustment_never_lowers_a_p_value(adjustment: PValueAdjustment) -> None:
    p_values = np.random.default_rng(1).uniform(size=20)
    adjusted = adjustment.adjust(p_values)
    assert np.all(adjusted >= p_values)
    assert np.all(adjusted <= 1)


@pytest.mark.parametrize("p_values", [[0.1, 1.5], [-0.1], [np.nan]])
def test_p_value_adjustment_rejects_values_outside_the_unit_interval(
    p_values: list[float],
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PValueAdjustment.HOLM.adjust(p_values)


def test_p_value_adjustment_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        PValueAdjustment.HOLM.adjust([[0.1, 0.2]])
    with pytest.raises(TypeError, match="numeric"):
        PValueAdjustment.HOLM.adjust([True, False])
