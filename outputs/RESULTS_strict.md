# Results — allosteric vs orthosteric modulator classification

*ChEMBL 37 · strict negatives only · benchmark run in 1403s on node5 (Linux x86_64, 24 cores)*

## Dataset

- **7,037 compounds** — 3,672 allosteric, 3,365 orthosteric (52.2% positive)
- **32 targets**, each carrying at least 5 compounds of *both* classes
- **1,900 generic Bemis–Murcko scaffolds**

Label tiers:

| tier               |   compounds |
|:-------------------|------------:|
| allosteric         |        3672 |
| orthosteric_strict |        3365 |

## Benchmark

**Mean per-fold ROC-AUC**, with a 95% bootstrap CI resampled *within* folds, plus the spread across folds. Metrics are computed inside each fold and then averaged: pooling out-of-fold predictions into a single ROC compares scores from models that never shared a scale (see *Pooling artefact*).

`target identity` uses only a one-hot of the primary target and carries no chemistry — it measures how much of the task is a target look-up.

| split    | features    | model    | fold ROC-AUC [95% CI]   |   fold SD |   pooled |    MCC |    n |
|:---------|:------------|:---------|:------------------------|----------:|---------:|-------:|-----:|
| random   | target_only | logistic | 0.950 [0.946–0.955]     |     0.006 |    0.949 |  0.75  | 7037 |
| random   | physchem    | lightgbm | 0.974 [0.971–0.977]     |     0.003 |    0.974 |  0.832 | 7037 |
| random   | physchem    | logistic | 0.841 [0.833–0.850]     |     0.011 |    0.841 |  0.522 | 7037 |
| random   | ecfp        | lightgbm | 0.996 [0.995–0.997]     |     0.001 |    0.996 |  0.952 | 7037 |
| random   | ecfp        | logistic | 0.990 [0.987–0.992]     |     0.001 |    0.99  |  0.924 | 7037 |
| random   | combined    | lightgbm | 0.995 [0.994–0.997]     |     0.002 |    0.995 |  0.953 | 7037 |
| random   | combined    | logistic | 0.990 [0.987–0.992]     |     0.002 |    0.99  |  0.924 | 7037 |
| scaffold | target_only | logistic | 0.932 [0.926–0.938]     |     0.028 |    0.935 |  0.738 | 7037 |
| scaffold | physchem    | lightgbm | 0.923 [0.916–0.929]     |     0.012 |    0.927 |  0.688 | 7037 |
| scaffold | physchem    | logistic | 0.791 [0.780–0.802]     |     0.043 |    0.796 |  0.434 | 7037 |
| scaffold | ecfp        | lightgbm | 0.983 [0.980–0.986]     |     0.009 |    0.986 |  0.881 | 7037 |
| scaffold | ecfp        | logistic | 0.969 [0.965–0.973]     |     0.014 |    0.972 |  0.857 | 7037 |
| scaffold | combined    | lightgbm | 0.983 [0.980–0.986]     |     0.009 |    0.986 |  0.882 | 7037 |
| scaffold | combined    | logistic | 0.969 [0.965–0.973]     |     0.014 |    0.972 |  0.855 | 7037 |
| temporal | target_only | logistic | 0.470 [0.431–0.507]     |     0     |    0.47  |  0.055 | 1309 |
| temporal | physchem    | lightgbm | 0.742 [0.715–0.768]     |     0     |    0.742 |  0.342 | 1309 |
| temporal | physchem    | logistic | 0.561 [0.527–0.595]     |     0     |    0.561 |  0.16  | 1309 |
| temporal | ecfp        | lightgbm | 0.820 [0.797–0.842]     |     0     |    0.82  |  0.419 | 1309 |
| temporal | ecfp        | logistic | 0.820 [0.796–0.844]     |     0     |    0.82  |  0.425 | 1309 |
| temporal | combined    | lightgbm | 0.829 [0.807–0.849]     |     0     |    0.829 |  0.354 | 1309 |
| temporal | combined    | logistic | 0.819 [0.795–0.842]     |     0     |    0.819 |  0.42  | 1309 |
| target   | target_only | logistic | 0.500 [0.500–0.500]     |     0     |    0.335 | -0.374 | 7037 |
| target   | physchem    | lightgbm | 0.623 [0.599–0.645]     |     0.17  |    0.686 |  0.272 | 7037 |
| target   | physchem    | logistic | 0.608 [0.581–0.632]     |     0.155 |    0.678 |  0.274 | 7037 |
| target   | ecfp        | lightgbm | 0.644 [0.617–0.672]     |     0.175 |    0.757 |  0.345 | 7037 |
| target   | ecfp        | logistic | 0.697 [0.670–0.722]     |     0.074 |    0.722 |  0.272 | 7037 |
| target   | combined    | lightgbm | 0.723 [0.697–0.748]     |     0.115 |    0.783 |  0.351 | 7037 |
| target   | combined    | logistic | 0.695 [0.670–0.720]     |     0.074 |    0.718 |  0.272 | 7037 |

### Two uncertainties, not one

The bootstrap CI above resamples compounds *within* each fold, so it measures within-fold sampling noise: **0.723 [0.697–0.748]**. A claim about generalising to a *new target* must instead carry the spread between held-out target groups, which is much wider: per-fold AUCs [0.5537, 0.8557, 0.8482, 0.6871, 0.6712], giving **0.723 [0.564–0.883]** (Student-t across 5 folds). The wider interval is the honest one for the headline claim.


*The temporal split is a single chronological holdout, so its `fold SD` is 0 by construction — one fold, not zero variability.*


### Pooling artefact

The target-identity probe is the worked example. Under the target-out split its one-hot features are all zero for a held-out target, so every prediction inside a fold is the same constant and the per-fold AUC is exactly **0.500** — chance, as it must be. Pooling those per-fold constants into one ROC instead reports **0.335**: an ordering invented between folds, not a below-chance model. Every headline number here is a fold-mean.


## Chemical space: pooled vs within-target

Median(allosteric) − median(orthosteric) for each property, computed two ways. **pooled** compares the two classes across the whole set and so confounds binding mode with which proteins each class came from. **within-target** averages the difference over targets that carry compounds of both classes, which is the only version that is about binding mode. A `sign_flip` is a Simpson's paradox: the pooled answer is the opposite of the real one.

| property       |   pooled Δ | within-target Δ [95% CI]   |   n_targets | flips   |
|:---------------|-----------:|:---------------------------|------------:|:--------|
| FractionCSP3   |      -0.08 | -0.08 [-0.12–-0.04]        |          32 |         |
| NumSpiroAtoms  |       0    | -0.06 [-0.15–+0.02]        |          32 |         |
| cLogP          |      -0.2  | +0.65 [-0.05–+1.35]        |          32 | **yes** |
| HBD            |      -1    | -0.53 [-0.98–-0.09]        |          32 |         |
| AliphaticRings |       0    | -0.34 [-0.57–-0.11]        |          32 |         |
| AromaticRings  |       0    | +0.31 [-0.10–+0.73]        |          32 |         |
| StereoCentres  |       0    | -0.38 [-0.86–+0.11]        |          32 |         |
| NumBridgeheads |       0    | -0.12 [-0.30–+0.05]        |          32 |         |
| RingCount      |       0    | -0.23 [-0.69–+0.22]        |          32 |         |
| BertzCT        |     -23.54 | +60.36 [-72.31–+193.04]    |          32 | **yes** |
| TPSA           |     -12.55 | -4.74 [-12.54–+3.05]       |          32 |         |
| RotB           |      -1    | -0.27 [-1.40–+0.87]        |          32 |         |

## Applicability domain

Maximum ECFP4 Tanimoto from each test compound to the training set. This is the measurement behind the split-scheme gap: a harder split is only harder because it moves test molecules further from what the model has seen.

| split    |   median_nn_tanimoto |   mean_nn_tanimoto |   frac_above_0.7 |   frac_above_0.4 |   n_test_compounds |
|:---------|---------------------:|-------------------:|-----------------:|-----------------:|-------------------:|
| random   |             0.784314 |           0.725282 |        0.720335  |         0.884894 |               7037 |
| scaffold |             0.660714 |           0.620407 |        0.372602  |         0.839278 |               7037 |
| temporal |             0.449438 |           0.458844 |        0.0763942 |         0.631016 |               1309 |
| target   |             0.295455 |           0.330183 |        0.0302686 |         0.149353 |               7037 |

## Head-to-head (paired bootstrap on identical resamples)

Differences between fold-mean AUCs, bootstrapped within folds on identical resamples for both models.

| split    | question                         | Δ fold ROC-AUC [95% CI]   | p        | significant   |
|:---------|:---------------------------------|:--------------------------|:---------|:--------------|
| random   | structure beyond physchem        | +0.022 [+0.019–+0.024]    | < 0.0005 | **yes**       |
| random   | chemistry beyond target identity | +0.045 [+0.041–+0.049]    | < 0.0005 | **yes**       |
| random   | fingerprint beyond physchem      | +0.022 [+0.019–+0.025]    | < 0.0005 | **yes**       |
| scaffold | structure beyond physchem        | +0.061 [+0.055–+0.066]    | < 0.0005 | **yes**       |
| scaffold | chemistry beyond target identity | +0.051 [+0.045–+0.057]    | < 0.0005 | **yes**       |
| scaffold | fingerprint beyond physchem      | +0.061 [+0.055–+0.067]    | < 0.0005 | **yes**       |
| temporal | structure beyond physchem        | +0.087 [+0.061–+0.114]    | < 0.0005 | **yes**       |
| temporal | chemistry beyond target identity | +0.358 [+0.321–+0.399]    | < 0.0005 | **yes**       |
| temporal | fingerprint beyond physchem      | +0.078 [+0.050–+0.110]    | < 0.0005 | **yes**       |
| target   | structure beyond physchem        | +0.100 [+0.077–+0.124]    | < 0.0005 | **yes**       |
| target   | fingerprint beyond physchem      | +0.021 [-0.009–+0.054]    | 0.195    | no            |

*`p` is a bootstrap tail fraction over 2000 resamples; `< 0.0005` means no resample crossed zero, not p = 0.*


## What the model keys on

Mean |SHAP| over out-of-fold rows under the target-out split.

`direction` is the mean signed SHAP over molecules where the feature is actually **present** — averaging over all rows would describe the feature being absent and inverts the sign for rare bits.

| feature      |   mean_abs_shap |   mean_signed_shap_when_present |   n_molecules_with_feature | direction      |
|:-------------|----------------:|--------------------------------:|---------------------------:|:---------------|
| ECFP4_1455   |        0.56211  |                       0.616085  |                       1573 | -> allosteric  |
| ECFP4_45     |        0.503107 |                       0.402057  |                       1700 | -> allosteric  |
| FractionCSP3 |        0.428953 |                       0.113959  |                       6781 | -> allosteric  |
| ECFP4_464    |        0.376612 |                       0.923305  |                       1466 | -> allosteric  |
| QED          |        0.278807 |                      -0.0772405 |                       7037 | -> orthosteric |
| ECFP4_1840   |        0.264685 |                       0.71204   |                       1554 | -> allosteric  |
| ECFP4_1825   |        0.228581 |                       0.441636  |                       1104 | -> allosteric  |
| cLogP        |        0.227185 |                      -0.0342147 |                       7010 | -> orthosteric |
| RotB         |        0.217918 |                      -0.0364668 |                       7010 | -> orthosteric |
| ECFP4_1928   |        0.215573 |                       0.13503   |                       3174 | -> allosteric  |
| ECFP4_1487   |        0.194546 |                       0.148453  |                       1492 | -> allosteric  |
| HBA          |        0.194424 |                      -0.0466399 |                       7035 | -> orthosteric |
| ECFP4_1452   |        0.191085 |                       0.125309  |                       3954 | -> allosteric  |
| HBD          |        0.186078 |                      -0.169799  |                       4911 | -> orthosteric |
| ECFP4_490    |        0.184068 |                       0.0284558 |                       1549 | -> allosteric  |
| TPSA         |        0.176026 |                      -0.0440034 |                       7036 | -> orthosteric |
| ECFP4_1816   |        0.173745 |                      -0.533027  |                       1440 | -> orthosteric |
| ECFP4_74     |        0.173744 |                      -0.40305   |                       1084 | -> orthosteric |
| ECFP4_1535   |        0.168692 |                       0.113056  |                       2564 | -> allosteric  |
| ECFP4_486    |        0.165215 |                      -0.837026  |                       1170 | -> orthosteric |

Top ECFP4 bits, class prevalence:

|   bit |   frac_in_allosteric |   frac_in_orthosteric |   enrichment |   n_molecules | example_smiles                                                     | example_substructures    |
|------:|---------------------:|----------------------:|-------------:|--------------:|:-------------------------------------------------------------------|:-------------------------|
|  1455 |             0.394063 |             0.0374443 |     10.524   |          1573 | CCCCCn1cc(-c2nc(C34CC5CC(CC(C5)C3)C4)no2)c(=O)c2cc(-c3ccco3)ccc21  | cn(c)C; cn(c)C; cn(-c)n  |
|    45 |             0.389434 |             0.0802377 |      4.8535  |          1700 | Cn1ncc2cc(C(=O)Nc3ccc(Cl)c(Cl)c3)ccc21                             | cc(c)n; cc(c)n; cc(c)n   |
|   464 |             0.360294 |             0.0424963 |      8.47825 |          1466 | CCCCCn1cc(-c2nc(C34CC5CC(CC(C5)C3)C4)no2)c(=O)c2cc(-c3ccco3)ccc21  | ccn; ccn; ccn            |
|  1840 |             0.35866  |             0.0704309 |      5.09237 |          1554 | Cn1ncc(Cl)c1-c1cc(C(=O)N[C@H](CN)Cc2ccc(F)c(F)c2)oc1Cl             | cc(c)F; CCN; CCN         |
|  1825 |             0.229847 |             0.077266  |      2.97476 |          1104 | O=C(CCCN1CCC2(CC1)C(=O)NCN2c1ccccc1)c1ccc(F)cc1                    | cc(c)N; cc(c)N; cc(c)N   |
|  1928 |             0.618464 |             0.268351  |      2.30469 |          3174 | O=C(CCCN1CCC2(CC1)C(=O)NCN2c1ccccc1)c1ccc(F)cc1                    | cF; cF; cF               |
|  1487 |             0.366558 |             0.0433878 |      8.4484  |          1492 | CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O | CC(C)O; ccc(F)cc; CC(C)O |
|  1452 |             0.690632 |             0.421397  |      1.63891 |          3954 | CNC(=O)[C@H]1O[C@@H](n2cnc3c(NCc4cccc(I)c4)ncnc32)[C@H](O)[C@@H]1O | cnc; COc; cnc            |

## Figures

![ROC-AUC under the three split schemes](figures/fig_split_gap.png)

![Signal by feature block](figures/fig_feature_blocks.png)

![ROC curves, combined features](figures/fig_roc.png)

![Pooled vs within-target property differences](figures/fig_property_paradox.png)

![Physicochemical property distributions (pooled)](figures/fig_physchem.png)

![Top SHAP features](figures/fig_shap.png)

![Class balance per target](figures/fig_target_composition.png)
