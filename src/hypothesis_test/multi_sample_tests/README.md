# multi_sample_tests

## Overview

Concrete `MultiSampleTest`s: tests that compare the location or the spread of several samples.

## Components

| Component | Description |
|-----------|-------------|
| [one_way_anova.py](./one_way_anova.py) | `OneWayAnova`, whether the means are equal, by Welch's or Fisher's F-test. |
| [alexander_govern.py](./alexander_govern.py) | `AlexanderGovern`, whether the means are equal, for unequal variances. |
| [kruskal_wallis.py](./kruskal_wallis.py) | `KruskalWallis`, whether the samples come from one distribution, from ranks. |
| [friedman.py](./friedman.py) | `Friedman`, whether repeated measures come from one distribution, from ranks within subjects. |
| [levene.py](./levene.py) | `Levene`, whether the variances are equal, from deviations from the center. |
| [bartlett.py](./bartlett.py) | `Bartlett`, whether the variances are equal, for normal data. |
| [fligner_killeen.py](./fligner_killeen.py) | `FlignerKilleen`, whether the variances are equal, from ranks of deviations. |

## Examples

```python
from hypothesis_test import KruskalWallis, OneWayAnova

print(OneWayAnova().run(group_a, group_b, group_c).p_value)
print(KruskalWallis().run_grouped(table["score"], table["group"]).p_value)
```
