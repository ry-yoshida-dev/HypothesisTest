# HypothesisTest

## Overview

HypothesisTest (`hypothesis_test`) is a Python package of statistical hypothesis tests behind one API. Each test is an object holding its options, and `run` returns the statistic, the p-value, the degrees of freedom, and a confidence interval where the test has one. Data the test is undefined for, such as a constant sample, gives an undefined result instead of a warning and NaN. It wraps SciPy, and computes from the published formula only what SciPy lacks.

| Group | Data | Tests |
| --- | --- | --- |
| [One sample](src/hypothesis_test/one_sample_tests/README.md) | One sample, or the differences of paired samples | `OneSampleT`, `WilcoxonSignedRank`, `Sign`, `ShapiroWilk`, `DAgostinoPearson`, `JarqueBera` |
| [Two samples](src/hypothesis_test/two_sample_tests/README.md) | Two independent samples | `TwoSampleT`, `MannWhitneyU`, `BrunnerMunzel`, `KolmogorovSmirnov`, `CramerVonMises` |
| [Multiple samples](src/hypothesis_test/multi_sample_tests/README.md) | Several samples, or repeated measures of the same subjects | `OneWayAnova`, `AlexanderGovern`, `KruskalWallis`, `Friedman`, `Levene`, `Bartlett`, `FlignerKilleen` |
| [Counts](src/hypothesis_test/count_tests/README.md) | Frequencies of categories or contingency tables | `ChiSquareGoodnessOfFit`, `ChiSquareIndependence`, `FisherExact`, `McNemar`, `Binomial` |

`PValueAdjustment` corrects the p-values of several tests for multiple comparisons by the Bonferroni, Holm, Benjamini-Hochberg, or Benjamini-Yekutieli method.

## Installation

```bash
uv add git+https://github.com/ry-yoshida-dev/HypothesisTest
```

## Usage

Run a test and read its result:

```python
from hypothesis_test import Alternative, TwoSampleT

result = TwoSampleT(alternative=Alternative.GREATER).run(treated, control)
print(result.statistic, result.p_value, result.degrees_of_freedom, result.confidence_interval)
print(result.is_significant(0.01))
```

Test paired samples through their differences, and samples in a long-format table by their labels:

```python
from hypothesis_test import KruskalWallis, WilcoxonSignedRank

print(WilcoxonSignedRank().run_paired(after, before).p_value)
print(KruskalWallis().run_grouped(scores, groups).p_value)
```

Test a contingency table, and adjust the p-values of several tests:

```python
from hypothesis_test import FisherExact, PValueAdjustment

print(FisherExact().run([[8, 2], [1, 5]]).confidence_interval)
print(PValueAdjustment.HOLM.adjust([0.01, 0.04, 0.03]))
```
