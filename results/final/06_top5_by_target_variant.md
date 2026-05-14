# Top-5 по (target, variant) — 4 группы

K = 5. Критериев: 6. ↑ — меньше лучше, ↓ — больше лучше.


## target=lt1 | variant=noabs  (n=41)


### raw_mae_min ↑

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0107     | EMG+NIRS+HRV  | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |



### kalman_mae_min ↑

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0107     | EMG+NIRS+HRV  | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0107     | EMG+NIRS+HRV  | ensemble |           nan |               nan |            nan |        3.7709 |           3.8112 | 0.78  | 0.166 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |



### r2 ↓

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2626 |           3.3112 | 0.792 | 0.584 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.6416 |           3.7108 | 0.728 | 0.489 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        3.9978 |           4.0737 | 0.736 | 0.444 |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | HRV           | linear   |           nan |               nan |            nan |        4.0095 |           4.1097 | 0.665 | 0.391 |                 nan |               nan |



## target=lt1 | variant=with_abs  (n=41)


### raw_mae_min ↑

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | HRV           | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |



### kalman_mae_min ↑

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |
| v0011     | HRV           | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |



### loso_mae_median ↑

_нет данных_


### loso_mae_std ↑ (стабильность)

_нет данных_


### rho ↓

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0107     | EMG+NIRS+HRV  | ensemble |           nan |               nan |            nan |        4.4935 |           4.5385 | 0.686 | 0.134 |                 nan |               nan |



### r2 ↓

| version   | feature_set   | family   |   inner_model |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|--------------:|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   |           nan |               nan |            nan |        3.2867 |           3.3515 | 0.791 | 0.574 |                 nan |               nan |
| v0011     | EMG+NIRS      | linear   |           nan |               nan |            nan |        3.5172 |           3.5886 | 0.745 | 0.52  |                 nan |               nan |
| v0011     | EMG           | linear   |           nan |               nan |            nan |        3.7667 |           3.8528 | 0.71  | 0.437 |                 nan |               nan |
| v0011     | HRV           | linear   |           nan |               nan |            nan |        4.0142 |           4.1424 | 0.654 | 0.379 |                 nan |               nan |
| v0011     | NIRS          | linear   |           nan |               nan |            nan |        4.1421 |           4.1327 | 0.719 | 0.371 |                 nan |               nan |



## target=lt2 | variant=noabs  (n=41)


### raw_mae_min ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 | 0.545 |               0.111 |                18 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |



### kalman_mae_min ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 | 0.545 |               0.111 |                18 |
| v0011     | NIRS          | linear   | GBM(n=50,d=2)    |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |



### loso_mae_median ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 | 0.369 |               0     |                17 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 | 0.545 |               0.111 |                18 |



### loso_mae_std ↑ (стабильность)

| version   | feature_set   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0102     | EMG+NIRS+HRV  | tcn      | nan           |            5.0861 |         1.24   |        5.466  |           5.6288 | 0.364 | -0.004 |               0.812 |                16 |
| v0011     | HRV           | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 |  0.68  |               0.056 |                18 |
| v0106b    | EMG+NIRS+HRV  | wavelet  | nan           |            4.3384 |         1.3346 |        4.5126 |           4.5712 | 0.524 |  0.313 |               0.471 |                17 |
| v0106b    | EMG+NIRS      | wavelet  | nan           |            4.8684 |         1.4086 |        4.8753 |           4.9134 | 0.445 |  0.213 |               0.588 |                17 |
| v0106b    | EMG           | wavelet  | nan           |            4.9822 |         1.4766 |        5.1461 |           5.1626 | 0.317 |  0.09  |               0.647 |                17 |



### rho ↓

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 |  0.584 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.5242 |         1.7883 |        2.9346 |           3.055  | 0.869 |  0.369 |               0     |                17 |
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 |  0.68  |               0.056 |                18 |
| v0107     | EMG+NIRS      | ensemble | ensemble         |            3.8489 |         2.1787 |        4.5283 |           4.5449 | 0.733 | -0.069 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 |  0.545 |               0.111 |                18 |



### r2 ↓

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.8409 |           2.9696 | 0.853 | 0.68  |               0.056 |                18 |
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.7187 |           2.8169 | 0.883 | 0.584 |               0.056 |                18 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.5644 |           3.6695 | 0.727 | 0.545 |               0.111 |                18 |
| v0011     | NIRS          | linear   | GBM(n=50,d=2)    |            3.1888 |         2.4574 |        3.874  |           3.8788 | 0.724 | 0.465 |               0.278 |                18 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |



## target=lt2 | variant=with_abs  (n=41)


### raw_mae_min ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 | 0.502 |               0.111 |                18 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |



### kalman_mae_min ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 | 0.502 |               0.111 |                18 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |



### loso_mae_median ↑

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 | 0.502 |               0.111 |                18 |



### loso_mae_std ↑ (стабильность)

| version   | feature_set   | family   | inner_model   |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:--------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0102     | EMG+NIRS+HRV  | tcn      | nan           |            4.8387 |         1.1636 |        5.0583 |           5.2227 | 0.523 | 0.12  |               0.438 |                16 |
| v0106b    | EMG+NIRS+HRV  | wavelet  | nan           |            4.0465 |         1.2639 |        4.33   |           4.4135 | 0.564 | 0.354 |               0.471 |                17 |
| v0011     | HRV           | linear   | Ridge(α=1000) |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0106b    | EMG           | wavelet  | nan           |            4.7364 |         1.3614 |        5.0247 |           5.0365 | 0.372 | 0.134 |               0.647 |                17 |
| v0106b    | EMG+NIRS      | wavelet  | nan           |            4.639  |         1.3805 |        4.8082 |           4.8532 | 0.46  | 0.221 |               0.588 |                17 |



### rho ↓

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |     r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|-------:|--------------------:|------------------:|
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 |  0.817 |               0.056 |                18 |
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 |  0.825 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 |  0.568 |               0     |                17 |
| v0104     | EMG+NIRS+HRV  | lstm     | nan              |            4.5187 |         1.8686 |        5.3917 |           5.6297 | 0.708 | -0.132 |               0.588 |                17 |
| v0106a    | EMG+NIRS+HRV  | wavelet  | nan              |            4.4698 |         1.8119 |        5.3905 |           5.6264 | 0.703 | -0.13  |               0.647 |                17 |



### r2 ↓

| version   | feature_set   | family   | inner_model      |   loso_mae_median |   loso_mae_std |   raw_mae_min |   kalman_mae_min |   rho |    r2 |   loso_neg_r2_share |   loso_n_subjects |
|:----------|:--------------|:---------|:-----------------|------------------:|---------------:|--------------:|-----------------:|------:|------:|--------------------:|------------------:|
| v0011     | EMG+NIRS+HRV  | linear   | EN(α=1.0,l1=0.9) |            1.8898 |         1.5453 |        2.2835 |           2.4817 | 0.909 | 0.825 |               0.056 |                18 |
| v0011     | HRV           | linear   | Ridge(α=1000)    |            1.8584 |         1.3271 |        2.2468 |           2.504  | 0.911 | 0.817 |               0.056 |                18 |
| v0107     | EMG+NIRS+HRV  | ensemble | ensemble         |            2.3839 |         1.9804 |        2.9603 |           3.1046 | 0.848 | 0.568 |               0     |                17 |
| v0011     | EMG+NIRS      | linear   | SVR(C=10,ε=1.0)  |            2.8805 |         2.2864 |        3.7412 |           3.8493 | 0.698 | 0.502 |               0.111 |                18 |
| v0011     | EMG           | linear   | SVR(C=10,ε=0.1)  |            2.6723 |         2.1105 |        3.8661 |           3.9506 | 0.669 | 0.419 |               0.111 |                18 |

