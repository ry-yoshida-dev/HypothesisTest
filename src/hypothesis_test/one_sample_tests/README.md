# one_sample_tests

## Overview

Concrete `OneSampleTest`s: tests of the location or the normality of one sample, which also test paired samples through their differences.

## Components

| Component | Description |
|-----------|-------------|
| [t.py](./t.py) | `OneSampleT`, whether the mean equals a value, under normality. |
| [wilcoxon_signed_rank.py](./wilcoxon_signed_rank.py) | `WilcoxonSignedRank`, whether a symmetric distribution is centered at a value. |
| [sign.py](./sign.py) | `Sign`, whether the median equals a value, for any distribution. |
| [shapiro_wilk.py](./shapiro_wilk.py) | `ShapiroWilk`, normality, for small and moderate samples. |
| [dagostino_pearson.py](./dagostino_pearson.py) | `DAgostinoPearson`, normality from skewness and kurtosis. |
| [jarque_bera.py](./jarque_bera.py) | `JarqueBera`, normality from skewness and kurtosis, for large samples. |

## Examples

```python
from hypothesis_test import Alternative, OneSampleT, ShapiroWilk

print(ShapiroWilk().run_paired(after, before).p_value)
print(OneSampleT(alternative=Alternative.GREATER).run_paired(after, before).p_value)
```
