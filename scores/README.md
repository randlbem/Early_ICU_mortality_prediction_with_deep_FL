# Early_ICU_mortality_prediction_with_deep_FL/scores

This folder houses the validation scores, test scores, and predictions for the tests we present in the thesis. Each pickle file in there relates to a single test run performed with five-fold cross-validation. The files are named according to the following convention:

  scores\_[*labeling*]\_[*number of clients*]\_[*history length*](*minimum ICU stay length*).pickle

- The folder *min24h* contains the results of the experiments on the cohort with $\Delta t_{min} = 24~h$
- The folder *min48h* contains the results of the experiments on the cohort with $\Delta t_{min} = 48~h$

## Results

The following tables contain the complete test-scores of the experiments performed for the paper.

![Table 1](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/scores/tables/table1.svg)
**Table 1:** Early stopping with min. loss ( $\Delta t_{min} = 24~h$ ).

![Table 2](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/scores/tables/table2.svg)
**Table 2:** early stopping with max. F1 ( $\Delta t_{min} = 24~h$ ).

![Table 3](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/scores/tables/table3.svg)
**Table 3:** early stopping with min. loss ( $\Delta t_{min} = 48~h$ ).

## Disclaimer

*This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.*

*This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.*

*You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.*
