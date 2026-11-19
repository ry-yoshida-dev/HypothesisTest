import numpy as np
import pytest
from scipy import stats

from hypothesis_test import (
    AlexanderGovern,
    Bartlett,
    Center,
    FlignerKilleen,
    Friedman,
    HypothesisTestResult,
    KruskalWallis,
    Levene,
    MultiSampleTest,
    OneWayAnova,
    TwoSampleT,
)

GROUP_A: list[float] = [4.2, 5.1, 3.9, 4.8, 5.5, 4.4]
GROUP_B: list[float] = [5.9, 6.3, 5.2, 6.8, 6.1]
GROUP_C: list[float] = [4.9, 7.8, 3.1, 6.6, 5.0, 8.2, 2.7]
UNDEFINED = HypothesisTestResult(statistic=None)

ALL_TESTS: list[MultiSampleTest] = [
    OneWayAnova(),
    OneWayAnova(is_variance_equal=False),
    AlexanderGovern(),
    KruskalWallis(),
    Friedman(),
    Levene(),
    Bartlett(),
    FlignerKilleen(),
]


def test_one_way_anova_matches_the_formula() -> None:
    groups = [np.array(GROUP_A), np.array(GROUP_B), np.array(GROUP_C)]
    pooled = np.concatenate(groups)
    between: float = sum(g.size * (np.mean(g) - np.mean(pooled)) ** 2 for g in groups) / 2
    within: float = sum(np.sum((g - np.mean(g)) ** 2) for g in groups) / (pooled.size - 3)
    result = OneWayAnova().run(*groups)
    assert result.statistic == pytest.approx(between / within)
    assert result.degrees_of_freedom == (2.0, pooled.size - 3.0)


@pytest.mark.parametrize("is_variance_equal", [True, False])
def test_one_way_anova_of_two_groups_is_the_squared_t_test(is_variance_equal: bool) -> None:
    anova = OneWayAnova(is_variance_equal=is_variance_equal).run(GROUP_A, GROUP_C)
    t_test = TwoSampleT(is_variance_equal=is_variance_equal).run(GROUP_A, GROUP_C)
    assert t_test.statistic is not None
    assert anova.statistic == pytest.approx(t_test.statistic**2)
    assert anova.p_value == pytest.approx(t_test.p_value)
    assert anova.degrees_of_freedom == pytest.approx((1.0, *t_test.degrees_of_freedom))


def test_one_way_anova_welch_matches_scipy() -> None:
    result = OneWayAnova(is_variance_equal=False).run(GROUP_A, GROUP_B, GROUP_C)
    reference = stats.f_oneway(GROUP_A, GROUP_B, GROUP_C, equal_var=False)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))


def test_one_way_anova_welch_is_undefined_with_a_constant_group() -> None:
    assert OneWayAnova(is_variance_equal=False).run([1.0, 1.0], GROUP_B) == UNDEFINED
    assert OneWayAnova().run([1.0, 1.0], GROUP_B).statistic is not None


def test_alexander_govern_matches_scipy() -> None:
    result = AlexanderGovern().run(GROUP_A, GROUP_B, GROUP_C)
    reference = stats.alexandergovern(GROUP_A, GROUP_B, GROUP_C)
    assert result.statistic == pytest.approx(float(reference.statistic))
    assert result.p_value == pytest.approx(float(reference.pvalue))
    assert result.degrees_of_freedom == (2.0,)


def test_kruskal_wallis_matches_the_formula_without_ties() -> None:
    groups = [np.array(GROUP_A), np.array(GROUP_B), np.array([3.0, 7.0, 8.0])]
    ranks = stats.rankdata(np.concatenate(groups))
    num: int = ranks.size
    boundaries = np.cumsum([g.size for g in groups])[:-1]
    rank_sums = [float(np.sum(part)) for part in np.split(ranks, boundaries)]
    expected: float = 12 / (num * (num + 1)) * sum(
        total**2 / g.size for total, g in zip(rank_sums, groups, strict=True)
    ) - 3 * (num + 1)
    assert KruskalWallis().run(*groups).statistic == pytest.approx(expected)


def test_run_grouped_orders_samples_by_label() -> None:
    values = GROUP_B + GROUP_A + GROUP_C
    labels = ["b"] * len(GROUP_B) + ["a"] * len(GROUP_A) + ["c"] * len(GROUP_C)
    order = np.random.default_rng(0).permutation(len(values))
    shuffled_values = np.array(values)[order]
    shuffled_labels = np.array(labels)[order]
    assert OneWayAnova().run_grouped(shuffled_values, shuffled_labels) == OneWayAnova().run(
        GROUP_A, GROUP_B, GROUP_C
    )


def test_run_grouped_keeps_the_blocks_of_repeated_measures() -> None:
    blocks = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.5]])
    labels = np.tile(["x", "y", "z"], blocks.shape[0])
    assert Friedman().run_grouped(blocks.ravel(), labels) == Friedman().run(*blocks.T)


def test_run_grouped_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="shape"):
        KruskalWallis().run_grouped([1.0, 2.0, 3.0], ["a", "b"])


def test_friedman_matches_the_formula_without_ties() -> None:
    blocks = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.5]])
    num_blocks, num_groups = blocks.shape
    rank_sums = np.sum(stats.rankdata(blocks, axis=1), axis=0)
    expected: float = 12 / (num_blocks * num_groups * (num_groups + 1)) * float(
        np.sum(rank_sums**2)
    ) - 3 * num_blocks * (num_groups + 1)
    result = Friedman().run(*blocks.T)
    assert result.statistic == pytest.approx(expected)
    assert result.degrees_of_freedom == (2.0,)


def test_friedman_requires_three_aligned_measures() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        Friedman().run(GROUP_A, GROUP_A)
    with pytest.raises(ValueError, match="same length"):
        Friedman().run(GROUP_A, GROUP_B, GROUP_C)


@pytest.mark.parametrize("center", list(Center))
def test_levene_is_the_anova_of_absolute_deviations(center: Center) -> None:
    groups = [np.array(GROUP_A), np.array(GROUP_B), np.array(GROUP_C)]
    deviations = [
        np.abs(g - (np.mean(g) if center is Center.MEAN else np.median(g))) for g in groups
    ]
    result = Levene(center=center).run(*groups)
    reference = OneWayAnova().run(*deviations)
    assert result.statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.p_value)
    assert result.degrees_of_freedom == reference.degrees_of_freedom


def test_equal_variance_tests_detect_unequal_spread() -> None:
    rng = np.random.default_rng(0)
    narrow = rng.normal(0, 1, size=60)
    wide = rng.normal(0, 4, size=60)
    for test in [Levene(), Bartlett(), FlignerKilleen()]:
        assert test.run(narrow, wide).is_significant(0.001)


def test_bartlett_and_fligner_killeen_match_scipy() -> None:
    assert Bartlett().run(GROUP_A, GROUP_B, GROUP_C).p_value == pytest.approx(
        float(stats.bartlett(GROUP_A, GROUP_B, GROUP_C).pvalue)
    )
    assert FlignerKilleen(center=Center.MEAN).run(
        GROUP_A, GROUP_B, GROUP_C
    ).p_value == pytest.approx(
        float(stats.fligner(GROUP_A, GROUP_B, GROUP_C, center="mean").pvalue)
    )


@pytest.mark.parametrize("test", ALL_TESTS)
def test_multi_sample_tests_are_undefined_for_constant_samples(test: MultiSampleTest) -> None:
    assert test.run([1.0] * 4, [1.0] * 4, [1.0] * 4) == UNDEFINED


@pytest.mark.parametrize("test", [AlexanderGovern(), Bartlett()])
def test_multi_sample_tests_are_undefined_with_any_constant_group(test: MultiSampleTest) -> None:
    assert test.run([1.0] * 4, GROUP_A, GROUP_B) == UNDEFINED


@pytest.mark.parametrize("test", ALL_TESTS)
def test_multi_sample_tests_run_without_warnings_on_random_data(test: MultiSampleTest) -> None:
    rng = np.random.default_rng(1)
    result = test.run(*rng.normal(size=(3, 20)))
    assert result.p_value is not None
    assert 0 <= result.p_value <= 1


def test_multi_sample_tests_are_undefined_for_too_few_observations() -> None:
    assert OneWayAnova().run([1.0], GROUP_A) == UNDEFINED


def test_multi_sample_tests_reject_too_few_groups_and_invalid_samples() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        KruskalWallis().run(GROUP_A)
    with pytest.raises(ValueError, match="sample 1 must be 1-D"):
        KruskalWallis().run(GROUP_A, [GROUP_B])


def test_multi_sample_tests_reject_invalid_options() -> None:
    with pytest.raises(TypeError, match="center"):
        Levene(center="median")  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
