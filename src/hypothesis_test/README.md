# hypothesis_test

## Overview

Hypothesis tests behind one API: an abstract class per shape of data, the concrete tests, and the results, options, and adjustments they share.

## Components

| Component | Description |
|-----------|-------------|
| [array_types.py](./array_types.py) | Type aliases of the arrays that tests compute on. |
| [parameter_validator.py](./parameter_validator.py) | `ParameterValidator`, runtime checks of test options. |
| [data_validator.py](./data_validator.py) | `DataValidator`, the conversion of samples and frequencies into validated arrays. |
| [options/](./options/README.md) | Enumerations that tests take as options. |
| [confidence_interval.py](./confidence_interval.py) | `ConfidenceInterval`. |
| [significance_outcome.py](./significance_outcome.py) | `SignificanceOutcome`, the result shape of backing SciPy tests. |
| [result.py](./result.py) | `HypothesisTestResult`, the outcome of a test. |
| [t_distribution.py](./t_distribution.py) | `TDistribution`, p-values and confidence intervals of t statistics. |
| [normal_distribution.py](./normal_distribution.py) | `NormalDistribution`, p-values of z statistics. |
| [monte_carlo_p_value.py](./monte_carlo_p_value.py) | `MonteCarloPValue`, p-values of statistics against statistics simulated under the null hypothesis. |
| [p_value_adjustment.py](./p_value_adjustment.py) | `PValueAdjustment`, corrections of p-values for multiple comparisons. |
| [one_sample_test.py](./one_sample_test.py) | `OneSampleTest`, the abstract test of one sample or of paired differences. |
| [one_sample_tests/](./one_sample_tests/README.md) | Concrete one-sample tests. |
| [two_sample_test.py](./two_sample_test.py) | `TwoSampleTest`, the abstract test of two independent samples. |
| [two_sample_tests/](./two_sample_tests/README.md) | Concrete two-sample tests. |
| [multi_sample_test.py](./multi_sample_test.py) | `MultiSampleTest`, the abstract test of several samples. |
| [multi_sample_tests/](./multi_sample_tests/README.md) | Concrete multi-sample tests. |
| [count_test.py](./count_test.py) | `CountTest`, the abstract test of frequencies. |
| [count_tests/](./count_tests/README.md) | Concrete count tests. |
