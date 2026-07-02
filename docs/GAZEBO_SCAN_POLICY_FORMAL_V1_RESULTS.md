# Gazebo Scan+Goal Policy Formal V1 Results

## Run Design

This run is the first formal Gazebo-derived perception-policy dataset for the AMR reliability project.

Input to policy:

- Gazebo lidar `/scan`, downsampled into fixed `scan_bin_*` features.
- Goal and navigation context such as robot pose, target pose, relative goal vector, risk score, localization uncertainty, sensor confidence, blockage score, obstacle proximity, and trajectory deviation.

Expert label:

- `expert_proxy_action` derived from Nav2 `/plan`.

Run matrix:

- seeds: `10`, `16`, `18`
- split rule: `10 -> train`, `16 -> val`, `18 -> test`
- goals: `east_south`, `west_near`, `north_near`, `south_axis`
- scenarios: `nominal`, `external_path_blockage`, `perception_degradation`
- total episodes: `36/36 ok`

Output directories:

- dataset logs: `outputs/gazebo_scan_policy_formal_v1`
- trained policy evidence: `outputs/gazebo_scan_policy_formal_train_v1`

## Action Imbalance

| split | EAST | NORTH | SOUTH | WEST |
|---|---:|---:|---:|---:|
| train | 69 / 7.89% | 42 / 4.81% | 385 / 44.05% | 378 / 43.25% |
| val | 58 / 7.18% | 37 / 4.58% | 366 / 45.30% | 347 / 42.95% |
| test | 65 / 7.82% | 41 / 4.93% | 374 / 45.01% | 351 / 42.24% |

Confirmed issue:

- The dataset now covers four movement actions.
- `NORTH` and `EAST` are minority classes.
- `STAY` is absent from Nav2-plan labels and should not be treated as a normal movement-policy class in this dataset.

## Policy Metrics

| model | split | rows | accuracy | macro F1 | weighted F1 | ECE | high-conf errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_scan_goal | train | 874 | 0.9989 | 0.7983 | 0.9989 | 0.0095 | 0 |
| baseline_scan_goal | val | 808 | 0.9035 | 0.6929 | 0.9167 | 0.0883 | 69 |
| baseline_scan_goal | test | 831 | 0.9819 | 0.7673 | 0.9817 | 0.0070 | 5 |
| class_weighted_focal_scan_goal | train | 874 | 0.9771 | 0.7606 | 0.9779 | 0.0296 | 0 |
| class_weighted_focal_scan_goal | val | 808 | 0.9777 | 0.7588 | 0.9786 | 0.0442 | 0 |
| class_weighted_focal_scan_goal | test | 831 | 0.9711 | 0.7474 | 0.9718 | 0.0255 | 4 |

Interpretation:

- The focal model strongly improves validation error behavior, especially high-confidence errors.
- On the held-out test seed, the baseline has better accuracy, macro F1, weighted F1, and ECE.
- Therefore focal loss is a candidate upgrade, not a confirmed global improvement.

## Class Recall

Baseline test recall:

| action | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| NORTH | 41 | 1.0000 | 0.8780 | 0.9351 |
| SOUTH | 374 | 0.9840 | 0.9893 | 0.9867 |
| EAST | 65 | 0.9365 | 0.9077 | 0.9219 |
| WEST | 351 | 0.9860 | 1.0000 | 0.9929 |

Focal test recall:

| action | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| NORTH | 41 | 0.8043 | 0.9024 | 0.8506 |
| SOUTH | 374 | 0.9945 | 0.9733 | 0.9838 |
| EAST | 65 | 0.8750 | 0.9692 | 0.9197 |
| WEST | 351 | 0.9885 | 0.9772 | 0.9828 |

Mechanism-relevant pattern:

- Focal improves minority `EAST` recall on test but reduces precision and overall calibration.
- Baseline misses some `NORTH` as `WEST` under perception degradation.
- Baseline validation failure is dominated by `SOUTH -> EAST` under external path blockage.

## High-Confidence Errors

High-confidence error patterns:

| model | split | scenario | actual -> predicted | count |
|---|---|---|---|---:|
| baseline_scan_goal | val | external_path_blockage | SOUTH -> EAST | 68 |
| baseline_scan_goal | val | perception_degradation | SOUTH -> EAST | 1 |
| baseline_scan_goal | test | perception_degradation | NORTH -> WEST | 5 |
| class_weighted_focal_scan_goal | test | perception_degradation | NORTH -> WEST | 4 |

This is the strongest residual-error evidence so far:

- External blockage can cause a high-confidence directional mistake in the baseline.
- Perception degradation can cause a high-confidence `NORTH -> WEST` confusion in both models.
- These residual errors are candidates for recovery-route design, but route rules should not be finalized until the mechanism analysis is broadened to more seeds and more disturbance types.

## Confirmed Facts

- Gazebo/Nav2 produced `36/36` successful episodes.
- The scan+goal dataset contains train/val/test splits.
- Four movement actions are represented in every split.
- The model can learn `scan + goal context -> Nav2-plan action`.
- Calibration and high-confidence error evidence is now available.

## Limitations

- There is still only one seed per split.
- `STAY` is absent because Nav2 `/plan` provides movement directions, not safety-stop labels.
- The current scenarios cover normal, blockage, and sensor degradation, but not yet localization drift or execution deviation in this formal scan-policy run.
- The results are simulation-derived and should not be framed as real-robot visual validation.

## Next Step

The next research step is residual mechanism analysis:

1. inspect the high-confidence error rows;
2. compare baseline vs focal embeddings/confidence under `external_path_blockage` and `perception_degradation`;
3. decide whether recovery routing should trigger on blockage-induced `SOUTH -> EAST`, perception-induced `NORTH -> WEST`, or both;
4. only then build the recovery-route mechanism from policy residual evidence.
