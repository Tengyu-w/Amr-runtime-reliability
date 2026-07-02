# Gazebo-Native Policy Analysis Results

This note records the corrected policy-analysis stage before recovery routing.
The policy is trained from Gazebo/Nav2 observer data, using Nav2 `/plan`-derived expert actions.

This is simulation evidence only. It is not a real-robot safety claim.

## Purpose

The goal is to follow the intended research order:

1. Collect Gazebo/Nav2 expert-action data.
2. Train a baseline policy.
3. Analyze imbalance, recall, confusion, calibration, and high-confidence errors.
4. Try a model upgrade.
5. Decide whether the upgrade genuinely improves the policy.
6. Only later use remaining policy failure mechanisms for recovery routing.

## Data

Source:

`AMR-Runtime-Reliability-Demo/outputs/gazebo_policy_monitor_matrix_v2/policy_evidence/policy_timesteps.csv`

Training output:

`AMR-Runtime-Reliability-Demo/outputs/gazebo_native_policy_v1`

Prepared trainable rows:

- Total evaluable Nav2-plan rows: 4,758
- Train rows: 2,134
- Val rows: 1,303
- Test rows: 1,321

Only rows with `policy_evaluable=True` and `expert_source=nav2_plan` were used.
The label is `expert_proxy_action` from the Nav2 plan direction.

## Imbalance

| Split | EAST | SOUTH | EAST Fraction | SOUTH Fraction |
| --- | ---: | ---: | ---: | ---: |
| train | 493 | 1,641 | 0.2310 | 0.7690 |
| val | 333 | 970 | 0.2556 | 0.7444 |
| test | 321 | 1,000 | 0.2430 | 0.7570 |

The action space in this dataset contains only `EAST` and `SOUTH`; `SOUTH` dominates by roughly 3:1.
This means the current Gazebo expert-action collection still lacks turn/stop/reverse diversity.

## Models

Two policies were trained:

- `baseline`: ordinary cross-entropy MLP.
- `class_weighted_focal`: class-weighted focal-loss MLP.

Inputs include robot pose, target pose, risk features, and derived goal-delta features.
Scenario id, fault origin/family, and OOD labels are not model inputs.

## Overall Metrics

| Model | Split | Accuracy | Macro F1 | ECE 10-bin | High-Conf Errors |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | train | 0.9981 | 0.9974 | 0.0026 | 0 |
| baseline | val | 0.9931 | 0.9908 | 0.0062 | 6 |
| baseline | test | 0.9977 | 0.9969 | 0.0027 | 0 |
| class_weighted_focal | train | 0.9953 | 0.9935 | 0.0067 | 0 |
| class_weighted_focal | val | 0.9893 | 0.9859 | 0.0076 | 5 |
| class_weighted_focal | test | 0.9909 | 0.9878 | 0.0078 | 0 |

The baseline is better overall. The class-weighted focal upgrade is not a general improvement.

## Per-Class Test Recall

| Model | Action | Support | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | EAST | 321 | 0.9969 | 0.9938 | 0.9953 |
| baseline | SOUTH | 1,000 | 0.9980 | 0.9990 | 0.9985 |
| class_weighted_focal | EAST | 321 | 0.9640 | 1.0000 | 0.9817 |
| class_weighted_focal | SOUTH | 1,000 | 1.0000 | 0.9880 | 0.9940 |

The upgrade fixes all test `EAST` misses, but it creates more `SOUTH -> EAST` errors.
So it is only useful if the research priority is minority-action recall over overall policy fidelity.

## Test Confusion

Baseline:

| Actual | Pred EAST | Pred SOUTH |
| --- | ---: | ---: |
| EAST | 319 | 2 |
| SOUTH | 1 | 999 |

Class-weighted focal:

| Actual | Pred EAST | Pred SOUTH |
| --- | ---: | ---: |
| EAST | 321 | 0 |
| SOUTH | 12 | 988 |

## Scenario-Level Test Metrics

Baseline test scenario metrics were strong:

- `external_path_blockage`, `mixed_blockage_and_perception`, `mixed_drift_and_execution`, `nominal`, `perception_degradation`, `progress_blockage`, `execution_deviation`: 1.0000 accuracy.
- `boundary_weak_blockage`: 0.9929 accuracy.
- `localization_drift`: 0.9869 accuracy.

The main baseline weakness is not broad failure across perturbation families; it is a small set of localized action confusions.

## High-Confidence Errors

Baseline high-confidence errors:

- Train: 0
- Val: 6
- Test: 0

The baseline high-confidence errors are concentrated in validation:

- `execution_deviation`: `EAST -> SOUTH`, 5 rows, confidence 1.0.
- `perception_degradation`: `EAST -> SOUTH`, 1 row, confidence 0.9864.

These are the first credible policy-error mechanism candidates after Gazebo-native training.

## Interpretation

The Gazebo-native policy fixes the main problem found in the earlier observer run: the grid-world policy did not align with Gazebo/Nav2 geometry.
After retraining on Nav2-plan labels, nominal and most fault scenarios become highly accurate.

The class-imbalance upgrade is not automatically beneficial:

- It improves minority `EAST` recall.
- It worsens total accuracy, macro-F1, calibration, and majority `SOUTH` recall.

Therefore the upgrade should not be adopted as the default policy yet.
It should be treated as an ablation showing the recall/calibration tradeoff.

## What Comes Next

The next proper step is not recovery routing yet.
The next step is to improve the policy dataset and policy analysis:

1. Collect richer Gazebo expert actions so labels include `STAY`, `NORTH`, and `WEST`, not only `EAST/SOUTH`.
2. Add local trajectory curvature and short history features.
3. Analyze the validation high-confidence `EAST -> SOUTH` errors by embedding and trajectory context.
4. Try a calibrated model upgrade, such as temperature scaling or validation-calibrated focal loss.
5. Only after the upgraded policy is chosen should remaining model-specific errors be converted into recovery-route mechanisms.
