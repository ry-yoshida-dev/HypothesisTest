# two_sample_tests

## Overview

Concrete `TwoSampleTest`s: tests that compare the location or the whole distribution of two independent samples.

## Components

| Component | Description |
|-----------|-------------|
| [t.py](./t.py) | `TwoSampleT`, whether the means differ, by Welch's or Student's t-test. |
| [mann_whitney_u.py](./mann_whitney_u.py) | `MannWhitneyU`, whether one sample tends to be larger, for distributions of one shape. |
| [brunner_munzel.py](./brunner_munzel.py) | `BrunnerMunzel`, whether one sample tends to be larger, for any distributions. |
| [kolmogorov_smirnov.py](./kolmogorov_smirnov.py) | `KolmogorovSmirnov`, whether the distributions differ, from the largest gap of their distribution functions. |
| [cramer_von_mises.py](./cramer_von_mises.py) | `CramerVonMises`, whether the distributions differ, from all gaps of their distribution functions. |
