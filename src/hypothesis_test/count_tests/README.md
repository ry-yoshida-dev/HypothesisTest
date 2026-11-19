# count_tests

## Overview

Concrete `CountTest`s: tests of categorical data from the frequencies of each category or of each cell of a contingency table.

## Components

| Component | Description |
|-----------|-------------|
| [chi_square_goodness_of_fit.py](./chi_square_goodness_of_fit.py) | `ChiSquareGoodnessOfFit`, whether categories occur with given proportions. |
| [chi_square_independence.py](./chi_square_independence.py) | `ChiSquareIndependence`, whether the variables of a contingency table are independent. |
| [fisher_exact.py](./fisher_exact.py) | `FisherExact`, whether the odds ratio of a 2 x 2 table is one, exactly. |
| [mcnemar.py](./mcnemar.py) | `McNemar`, whether paired binary outcomes change in either direction equally often. |
| [binomial.py](./binomial.py) | `Binomial`, whether the probability of success equals a value. |

## Examples

```python
from hypothesis_test import Binomial, ChiSquareIndependence

print(ChiSquareIndependence().run([[12, 5], [7, 14]]).p_value)
print(Binomial(expected_proportion=0.3).run([18, 32]).confidence_interval)
```
