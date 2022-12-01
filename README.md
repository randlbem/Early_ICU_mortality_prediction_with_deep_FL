# Early_ICU_mortality_prediction_with_deep_FL

This repository houses the code used in our thesis, "Early prediction of the risk of ICU mortality with Deep Federated Learning". All code is our own apart from the notebook *mimic_iii_preprocessing.ipynb*, which we took and adapted from Mondrejevski et al. (2022) and is used to preprocess the raw data.

- The file *mimic_iii_windowing.ipynb* provides both window selection and labeling steps.
- The file *model.ipynb* was used for training and evaluating the models.
- The folder *analysis* contains jupyter-notebooks we used for analysing the data and creating plots and tables.
- The folder *assets* contains classes and functions needed for training and evaluating the model.
- The folder *scores* contains the results of our experiments in the form of pickle files.
- The folder *sql* contains sql-files for creating database-tables from MIMIC-III (v1.4).

## Abstract

Intensive Care Units usually carry patients with a serious risk of mortality. Recent research has shown the ability of Machine Learning to indicate the patients’ mortality risk and point physicians toward individuals with a heightened need for care. Nevertheless, healthcare data is often subject to privacy regulations and can therefore not be easily shared in order to build Centralized Machine Learning models that use the combined data of multiple hospitals. Federated Learning is a Machine Learning framework designed for data privacy that can be used to circumvent this problem. In this study, we evaluate the ability of deep Federated Learning to predict the risk of Intensive Care Unit mortality at an early stage. We compare the predictive performance of Federated, Centralized, and Local Machine Learning in terms of AUPRC, F1-score, and AUROC. Our results show that Federated Learning performs equally well as the centralized approach and is substantially better than the local approach, thus providing a viable solution for early Intensive Care Unit mortality prediction. In addition, we show that the prediction performance is higher when the patient history window is closer to discharge or death. Finally, we show that using the F1-score as an early stopping metric can stabilize and increase the performance of our approach for the task at hand.

**Keywords:** *Federated Learning* · *Early Mortality Prediction* · *Recurrent Neural Networks* · *Multivariate Time Series* · *Intensive Care Unit*

## Dataset

All the tests have been performed using the MIMIC-III (v1.4) dataset (Johnson et al., 2016). Access to this dataset can be gained through [PhysioNet](https://physionet.org/content/mimiciii/1.4/). This repository does not contain any data from the dataset.

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

*All rights to the files *[sql/flicu_icustay_detail.sql](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/sql/flicu_icustay_detail.sql)*, *[sql/flicu_pivoted_lab.sql](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/sql/flicu_pivoted_lab.sql)*, *[sql/pivoted_lab.sql](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/sql/pivoted_lab.sql)*, *[sql/pivoted_vital.sql](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/sql/flicu_pivoted_vital.sql)*, and *[sql/postgres-functions.sql](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/sql/postgres-functions.sql)* belong to their author Lena Mondrejevski. The file *[mimic_iii_preprocessing.ipynb](https://github.com/randlbem/Early_ICU_mortality_prediction_with_deep_FL/blob/main/mimic_iii_preprocessing.ipynb)* was also authored by Lena Mondrejevski but further altered by us.*

_____________________________________________________________________________________________
[Mondrejevski, L., Miliou, I., Montanino, A., Pitts, D., Hollmén, J. & Papapetrou, P. (2022),
‘FLICU: A federated learning workflow for intensive care unit mortality prediction’.](https://arxiv.org/abs/2205.15104 "FLICU: A federated learning workflow for intensive care unit mortality prediction")

[Johnson, A. E. W., Pollard, T. J. & Mark, R. G. (2016), ‘Mimic-iii clinical database (version 1.4)’,
PhysioNet.](https://doi.org/10.13026/C2XW26 "Mimic-iii clinical database (version 1.4)")
