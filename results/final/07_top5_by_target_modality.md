# Top-5 по модальностям (target × feature_set), варианты видны как строки

K = 5. Критериев: 6. ↑ — меньше лучше, ↓ — больше лучше.


## target=lt1 | feature_set=EMG  (n=20)


### raw_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.1508 |           5.0756 | 0.387 | 0.106 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        5.2686 |           5.1909 | 0.484 | 0.064 |                 nan |               nan |
| v0104     | noabs     | lstm     |           nan |               nan |            nan |        5.2778 |           5.176  | 0.45  | 0.023 |                 nan |               nan |



### kalman_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.1508 |           5.0756 | 0.387 | 0.106 |                 nan |               nan |
| v0104     | noabs     | lstm     |           nan |               nan |            nan |        5.2778 |           5.176  | 0.45  | 0.023 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        5.2686 |           5.1909 | 0.484 | 0.064 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  |  0.437 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  |  0.437 |                 nan |               nan |
| v0101     | with_abs  | lstm     |           nan |               nan |            nan |        5.4044 |           5.256  | 0.58  | -0.037 |                 nan |               nan |
| v0101     | noabs     | lstm     |           nan |               nan |            nan |        5.4065 |           5.2568 | 0.555 | -0.039 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        5.2686 |           5.1909 | 0.484 |  0.064 |                 nan |               nan |



### r2 ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.1508 |           5.0756 | 0.387 | 0.106 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        5.2686 |           5.1909 | 0.484 | 0.064 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        5.3244 |           5.2299 | 0.277 | 0.057 |                 nan |               nan |



## target=lt1 | feature_set=EMG+NIRS  (n=20)


### raw_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9544 |           4.9066 | 0.435 | 0.155 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.0262 |           4.9823 | 0.429 | 0.145 |                 nan |               nan |
| v0104     | with_abs  | lstm     |           nan |               nan |            nan |        5.1385 |           5.0732 | 0.601 | 0.08  |                 nan |               nan |



### kalman_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9544 |           4.9066 | 0.435 | 0.155 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.0262 |           4.9823 | 0.429 | 0.145 |                 nan |               nan |
| v0104     | with_abs  | lstm     |           nan |               nan |            nan |        5.1385 |           5.0732 | 0.601 | 0.08  |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 |  0.52  |                 nan |               nan |
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 |  0.489 |                 nan |               nan |
| v0101     | with_abs  | lstm     |           nan |               nan |            nan |        5.3149 |           5.1811 | 0.665 |  0.012 |                 nan |               nan |
| v0101     | noabs     | lstm     |           nan |               nan |            nan |        5.3497 |           5.2123 | 0.633 | -0.004 |                 nan |               nan |
| v0104     | with_abs  | lstm     |           nan |               nan |            nan |        5.1385 |           5.0732 | 0.601 |  0.08  |                 nan |               nan |



### r2 ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9544 |           4.9066 | 0.435 | 0.155 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        5.0262 |           4.9823 | 0.429 | 0.145 |                 nan |               nan |
| v0104     | with_abs  | lstm     |           nan |               nan |            nan |        5.1385 |           5.0732 | 0.601 | 0.08  |                 nan |               nan |



## target=lt1 | feature_set=EMG+NIRS+HRV  (n=20)


### raw_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.4935 |           4.5385 | 0.686 | 0.134 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.5684 |           4.5663 | 0.585 | 0.269 |                 nan |               nan |



### kalman_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.4935 |           4.5385 | 0.686 | 0.134 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.5684 |           4.5663 | 0.585 | 0.269 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.4935 |           4.5385 | 0.686 | 0.134 |                 nan |               nan |
| v0101     | with_abs  | lstm     |           nan |               nan |            nan |        5.3341 |           5.2103 | 0.66  | 0.011 |                 nan |               nan |



### r2 ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0106b    | with_abs  | wavelet  |           nan |               nan |            nan |        4.5684 |           4.5663 | 0.585 | 0.269 |                 nan |               nan |
| v0106b    | noabs     | wavelet  |           nan |               nan |            nan |        4.6857 |           4.6813 | 0.574 | 0.239 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |



## target=lt1 | feature_set=HRV  (n=2)


### raw_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        4.0095 |           4.1097 | 0.665 | 0.391 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |



### kalman_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        4.0095 |           4.1097 | 0.665 | 0.391 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        4.0095 |           4.1097 | 0.665 | 0.391 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |



### r2 ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        4.0095 |           4.1097 | 0.665 | 0.391 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |



## target=lt1 | feature_set=NIRS  (n=20)


### raw_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        4.6622 |           4.7261 | 0.613 | 0.177 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.9151 |           4.9542 | 0.593 | 0.111 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9925 |           4.9702 | 0.662 | 0.126 |                 nan |               nan |



### kalman_mae_min ↑

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        4.6622 |           4.7261 | 0.613 | 0.177 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.9151 |           4.9542 | 0.593 | 0.111 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9925 |           4.9702 | 0.662 | 0.126 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 |  0.444 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 |  0.371 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9925 |           4.9702 | 0.662 |  0.126 |                 nan |               nan |
| v0101     | with_abs  | lstm     |           nan |               nan |            nan |        5.4365 |           5.2754 | 0.619 | -0.021 |                 nan |               nan |
| v0101     | noabs     | lstm     |           nan |               nan |            nan |        5.4396 |           5.293  | 0.615 | -0.026 |                 nan |               nan |



### r2 ↓

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |
| v0011     | with_abs  | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |
| v0107     | noabs     | ensemble |           nan |               nan |            nan |        4.6622 |           4.7261 | 0.613 | 0.177 |                 nan |               nan |
| v0106a    | with_abs  | wavelet  |           nan |               nan |            nan |        4.9925 |           4.9702 | 0.662 | 0.126 |                 nan |               nan |
| v0107     | with_abs  | ensemble |           nan |               nan |            nan |        4.9151 |           4.9542 | 0.593 | 0.111 |                 nan |               nan |



## target=lt2 | feature_set=EMG  (n=20)


### raw_mae_min ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0106b    | with_abs  | wavelet  | nan             |            4.7364 |         1.3614 |        5.0247 |           5.0365 | 0.372 |  0.134 |               0.647 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.9822 |         1.4766 |        5.1461 |           5.1626 | 0.317 |  0.09  |               0.647 |                17 |
| v0107     | with_abs  | ensemble | ensemble        |            3.8425 |         2.9898 |        5.3482 |           5.3113 | 0.538 | -0.913 |               0     |                17 |



### kalman_mae_min ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0106b    | with_abs  | wavelet  | nan             |            4.7364 |         1.3614 |        5.0247 |           5.0365 | 0.372 |  0.134 |               0.647 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.9822 |         1.4766 |        5.1461 |           5.1626 | 0.317 |  0.09  |               0.647 |                17 |
| v0107     | with_abs  | ensemble | ensemble        |            3.8425 |         2.9898 |        5.3482 |           5.3113 | 0.538 | -0.913 |               0     |                17 |



### loso_mae_median ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0107     | noabs     | ensemble | ensemble        |            3.7184 |         3.0048 |        5.357  |           5.3254 | 0.534 | -0.868 |               0     |                17 |
| v0107     | with_abs  | ensemble | ensemble        |            3.8425 |         2.9898 |        5.3482 |           5.3113 | 0.538 | -0.913 |               0     |                17 |
| v0106a    | noabs     | wavelet  | nan             |            4.4135 |         1.819  |        5.5576 |           5.7659 | 0.414 | -0.175 |               0.706 |                17 |



### loso_mae_std ↑ (стабильность)

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0106b    | with_abs  | wavelet  |           nan |            4.7364 |         1.3614 |        5.0247 |           5.0365 | 0.372 |  0.134 |               0.647 |                17 |
| v0106b    | noabs     | wavelet  |           nan |            4.9822 |         1.4766 |        5.1461 |           5.1626 | 0.317 |  0.09  |               0.647 |                17 |
| v0105     | noabs     | tcn      |           nan |            4.508  |         1.6169 |        5.3747 |           5.414  | 0.125 | -0.005 |               0.875 |                16 |
| v0105     | with_abs  | tcn      |           nan |            4.7038 |         1.6403 |        5.4056 |           5.4362 | 0.088 | -0.015 |               0.938 |                16 |
| v0106c    | with_abs  | wavelet  |           nan |            4.6965 |         1.7668 |        5.633  |           5.7986 | 0.446 | -0.182 |               0.765 |                17 |



### rho ↓

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0101     | noabs     | lstm     | nan             |            4.7562 |         2.029  |        5.9187 |           6.1408 | 0.557 | -0.317 |               0.833 |                18 |
| v0107     | with_abs  | ensemble | ensemble        |            3.8425 |         2.9898 |        5.3482 |           5.3113 | 0.538 | -0.913 |               0     |                17 |
| v0107     | noabs     | ensemble | ensemble        |            3.7184 |         3.0048 |        5.357  |           5.3254 | 0.534 | -0.868 |               0     |                17 |



### r2 ↓

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=0.1) |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 |  0.419 |               0.111 |                18 |
| v0106b    | with_abs  | wavelet  | nan             |            4.7364 |         1.3614 |        5.0247 |           5.0365 | 0.372 |  0.134 |               0.647 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.9822 |         1.4766 |        5.1461 |           5.1626 | 0.317 |  0.09  |               0.647 |                17 |
| v0105     | noabs     | tcn      | nan             |            4.508  |         1.6169 |        5.3747 |           5.414  | 0.125 | -0.005 |               0.875 |                16 |



## target=lt2 | feature_set=EMG+NIRS  (n=20)


### raw_mae_min ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 |  0.502 |               0.111 |                18 |
| v0107     | noabs     | ensemble | ensemble        |            3.8489 |         2.1787 |        4.5283 |           4.5449 | 0.733 | -0.069 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan             |            4.639  |         1.3805 |        4.8082 |           4.8532 | 0.46  |  0.221 |               0.588 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.8684 |         1.4086 |        4.8753 |           4.9134 | 0.445 |  0.213 |               0.588 |                17 |



### kalman_mae_min ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 |  0.502 |               0.111 |                18 |
| v0107     | noabs     | ensemble | ensemble        |            3.8489 |         2.1787 |        4.5283 |           4.5449 | 0.733 | -0.069 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan             |            4.639  |         1.3805 |        4.8082 |           4.8532 | 0.46  |  0.221 |               0.588 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.8684 |         1.4086 |        4.8753 |           4.9134 | 0.445 |  0.213 |               0.588 |                17 |



### loso_mae_median ↑

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 |  0.502 |               0.111 |                18 |
| v0107     | noabs     | ensemble | ensemble        |            3.8489 |         2.1787 |        4.5283 |           4.5449 | 0.733 | -0.069 |               0     |                17 |
| v0107     | with_abs  | ensemble | ensemble        |            4.2825 |         2.4679 |        5.1707 |           5.1461 | 0.615 | -0.255 |               0     |                17 |
| v0104     | with_abs  | lstm     | nan             |            4.3886 |         1.9528 |        5.6916 |           5.8974 | 0.385 | -0.217 |               0.765 |                17 |



### loso_mae_std ↑ (стабильность)

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0106b    | with_abs  | wavelet  |           nan |            4.639  |         1.3805 |        4.8082 |           4.8532 | 0.46  |  0.221 |               0.588 |                17 |
| v0106b    | noabs     | wavelet  |           nan |            4.8684 |         1.4086 |        4.8753 |           4.9134 | 0.445 |  0.213 |               0.588 |                17 |
| v0102     | noabs     | tcn      |           nan |            5.5145 |         1.5402 |        5.9054 |           6.0198 | 0.118 | -0.193 |               0.875 |                16 |
| v0105     | with_abs  | tcn      |           nan |            4.6189 |         1.568  |        5.3663 |           5.399  | 0.198 | -0.004 |               1     |                16 |
| v0105     | noabs     | tcn      |           nan |            4.8035 |         1.585  |        5.3987 |           5.4295 | 0.142 | -0.009 |               0.875 |                16 |



### rho ↓

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0107     | noabs     | ensemble | ensemble        |            3.8489 |         2.1787 |        4.5283 |           4.5449 | 0.733 | -0.069 |               0     |                17 |
| v0011     | noabs     | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 |  0.502 |               0.111 |                18 |
| v0107     | with_abs  | ensemble | ensemble        |            4.2825 |         2.4679 |        5.1707 |           5.1461 | 0.615 | -0.255 |               0     |                17 |
| v0101     | with_abs  | lstm     | nan             |            4.7691 |         2.0552 |        5.9353 |           6.1622 | 0.561 | -0.323 |               0.778 |                18 |



### r2 ↓

| version   | variant   | family   | inner_model     |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |
| v0011     | with_abs  | linear   | SVR(C=10,ε=1.0) |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 |  0.502 |               0.111 |                18 |
| v0106b    | with_abs  | wavelet  | nan             |            4.639  |         1.3805 |        4.8082 |           4.8532 | 0.46  |  0.221 |               0.588 |                17 |
| v0106b    | noabs     | wavelet  | nan             |            4.8684 |         1.4086 |        4.8753 |           4.9134 | 0.445 |  0.213 |               0.588 |                17 |
| v0105     | with_abs  | tcn      | nan             |            4.6189 |         1.568  |        5.3663 |           5.399  | 0.198 | -0.004 |               1     |                16 |



## target=lt2 | feature_set=EMG+NIRS+HRV  (n=20)


### raw_mae_min ↑

| version   | variant   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0011     | noabs     | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0107     | noabs     | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0107     | with_abs  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan              |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 | 0.354 |               0.471 |                17 |



### kalman_mae_min ↑

| version   | variant   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0011     | noabs     | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0107     | noabs     | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0107     | with_abs  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan              |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 | 0.354 |               0.471 |                17 |



### loso_mae_median ↑

| version   | variant   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0011     | with_abs  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0107     | with_abs  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0107     | noabs     | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan              |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 | 0.354 |               0.471 |                17 |



### loso_mae_std ↑ (стабильность)

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0102     | with_abs  | tcn      |           nan |            4.8387 |         1.1636 |        5.0583 |           5.2227 | 0.523 |  0.12  |               0.438 |                16 |
| v0102     | noabs     | tcn      |           nan |            5.0861 |         1.24   |        5.466  |           5.6288 | 0.364 | -0.004 |               0.812 |                16 |
| v0106b    | with_abs  | wavelet  |           nan |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 |  0.354 |               0.471 |                17 |
| v0106b    | noabs     | wavelet  |           nan |            4.3384 |         1.3346 |        4.5126 |           4.5712 | 0.524 |  0.313 |               0.471 |                17 |
| v0105     | with_abs  | tcn      |           nan |            4.6236 |         1.5195 |        5.2931 |           5.3266 | 0.342 |  0.029 |               0.938 |                16 |



### rho ↓

| version   | variant   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 |  0.825 |               0.056 |                18 |
| v0011     | noabs     | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 |  0.584 |               0.056 |                18 |
| v0107     | noabs     | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 |  0.369 |               0     |                17 |
| v0107     | with_abs  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 |  0.568 |               0     |                17 |
| v0104     | with_abs  | lstm     | nan              |            4.5187 |         1.8686 |        5.3917 |           5.6297 | 0.708 | -0.132 |               0.588 |                17 |



### r2 ↓

| version   | variant   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0011     | noabs     | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0107     | with_abs  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0107     | noabs     | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan              |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 | 0.354 |               0.471 |                17 |



## target=lt2 | feature_set=HRV  (n=2)


### raw_mae_min ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



### kalman_mae_min ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



### loso_mae_median ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



### loso_mae_std ↑ (стабильность)

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



### rho ↓

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



### r2 ↓

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | with_abs  | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | noabs     | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |



## target=lt2 | feature_set=NIRS  (n=20)


### raw_mae_min ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0011     | with_abs  | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        4.2199 |           4.2895 | 0.615 | 0.333 |               0.278 |                18 |
| v0107     | noabs     | ensemble | ensemble      |            3.807  |         2.3694 |        4.5032 |           4.5506 | 0.629 | 0.239 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan           |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 | 0.193 |               0.412 |                17 |
| v0106b    | noabs     | wavelet  | nan           |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  | 0.172 |               0.235 |                17 |



### kalman_mae_min ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0011     | with_abs  | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        4.2199 |           4.2895 | 0.615 | 0.333 |               0.278 |                18 |
| v0107     | noabs     | ensemble | ensemble      |            3.807  |         2.3694 |        4.5032 |           4.5506 | 0.629 | 0.239 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan           |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 | 0.193 |               0.412 |                17 |
| v0106b    | noabs     | wavelet  | nan           |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  | 0.172 |               0.235 |                17 |



### loso_mae_median ↑

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0011     | with_abs  | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        4.2199 |           4.2895 | 0.615 | 0.333 |               0.278 |                18 |
| v0106b    | noabs     | wavelet  | nan           |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  | 0.172 |               0.235 |                17 |
| v0107     | noabs     | ensemble | ensemble      |            3.807  |         2.3694 |        4.5032 |           4.5506 | 0.629 | 0.239 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan           |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 | 0.193 |               0.412 |                17 |



### loso_mae_std ↑ (стабильность)

| version   | variant   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0102     | noabs     | tcn      |           nan |            5.1376 |         1.5696 |        5.6766 |           5.8092 | 0.173 | -0.166 |               1     |                16 |
| v0105     | noabs     | tcn      |           nan |            4.6464 |         1.6018 |        5.3919 |           5.4525 | 0.235 | -0.024 |               0.938 |                16 |
| v0106b    | with_abs  | wavelet  |           nan |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 |  0.193 |               0.412 |                17 |
| v0106b    | noabs     | wavelet  |           nan |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  |  0.172 |               0.235 |                17 |
| v0105     | with_abs  | tcn      |           nan |            4.6299 |         1.7146 |        5.4491 |           5.5119 | 0.061 | -0.058 |               0.938 |                16 |



### rho ↓

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0107     | noabs     | ensemble | ensemble      |            3.807  |         2.3694 |        4.5032 |           4.5506 | 0.629 | 0.239 |               0     |                17 |
| v0106b    | noabs     | wavelet  | nan           |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  | 0.172 |               0.235 |                17 |
| v0011     | with_abs  | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        4.2199 |           4.2895 | 0.615 | 0.333 |               0.278 |                18 |
| v0106b    | with_abs  | wavelet  | nan           |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 | 0.193 |               0.412 |                17 |



### r2 ↓

| version   | variant   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:----------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | noabs     | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0011     | with_abs  | linear   | GBM(n=50,d=2) |            3.1888 |         2.4574 |        4.2199 |           4.2895 | 0.615 | 0.333 |               0.278 |                18 |
| v0107     | noabs     | ensemble | ensemble      |            3.807  |         2.3694 |        4.5032 |           4.5506 | 0.629 | 0.239 |               0     |                17 |
| v0106b    | with_abs  | wavelet  | nan           |            3.9291 |         1.6327 |        4.6416 |           4.7672 | 0.574 | 0.193 |               0.412 |                17 |
| v0106b    | noabs     | wavelet  | nan           |            3.7756 |         1.6528 |        4.7228 |           4.8574 | 0.62  | 0.172 |               0.235 |                17 |

