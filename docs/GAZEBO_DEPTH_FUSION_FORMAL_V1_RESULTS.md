# Gazebo Depth And Fusion Formal V1 Results

## Run Design

This run extends the scan-policy setup with a Gazebo depth camera.

Dataset:

- simulator: Gazebo + Nav2
- episodes: `36/36 ok`
- seeds: `10`, `16`, `18`
- split rule: `10 -> train`, `16 -> val`, `18 -> test`
- goals: `east_south`, `west_near`, `north_near`, `south_axis`
- scenarios: `nominal`, `external_path_blockage`, `perception_degradation`

Recorded modalities:

- scan-policy logs: planar lidar `/scan`
- depth-policy logs: forward depth image `/depth_image`, compressed into `depth_cell_*` grid features
- expert labels: `expert_proxy_action` from Nav2 `/plan`

Output directories:

- raw formal logs: `outputs/gazebo_depth_policy_formal_v1`
- scan-only training: `outputs/gazebo_scan_policy_depth_matrix_train_v2`
- depth-only training: `outputs/gazebo_depth_policy_formal_train_v2`
- scan+depth fusion training: `outputs/gazebo_fusion_policy_formal_train_v2`

## Action Distribution

| split | EAST | NORTH | SOUTH | WEST |
|---|---:|---:|---:|---:|
| train | 126 / 10.99% | 69 / 6.02% | 592 / 51.61% | 360 / 31.39% |
| val | 173 / 14.70% | 68 / 5.78% | 555 / 47.15% | 381 / 32.37% |
| test | 144 / 12.71% | 57 / 5.03% | 573 / 50.57% | 359 / 31.69% |

`STAY` is still absent because Nav2 `/plan` gives movement directions, not safety-stop labels.

## Modality Comparison

Deterministic training seed was fixed in the training scripts before these results were generated.

| modality | model | split | rows | accuracy | macro F1 | weighted F1 | ECE | high-conf errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| scan | baseline_scan_goal | val | 1177 | 0.8768 | 0.6830 | 0.8753 | 0.0619 | 63 |
| scan | baseline_scan_goal | test | 1133 | 0.8694 | 0.6741 | 0.8776 | 0.0778 | 78 |
| depth | baseline_depth_goal | val | 1177 | 0.8743 | 0.6796 | 0.8726 | 0.0663 | 55 |
| depth | baseline_depth_goal | test | 1133 | 0.8773 | 0.6666 | 0.8819 | 0.0553 | 62 |
| fusion | baseline_scan_depth_goal | val | 1177 | 0.8318 | 0.6323 | 0.8248 | 0.1022 | 89 |
| fusion | baseline_scan_depth_goal | test | 1133 | 0.8641 | 0.6559 | 0.8673 | 0.0721 | 64 |
| scan | class_weighted_focal_scan_goal | val | 1177 | 0.8496 | 0.6721 | 0.8556 | 0.0284 | 25 |
| scan | class_weighted_focal_scan_goal | test | 1133 | 0.8076 | 0.6343 | 0.8235 | 0.0617 | 51 |
| fusion | class_weighted_focal_scan_depth_goal | val | 1177 | 0.8224 | 0.6576 | 0.8334 | 0.0543 | 24 |
| fusion | class_weighted_focal_scan_depth_goal | test | 1133 | 0.8111 | 0.6318 | 0.8265 | 0.0549 | 52 |

## Interpretation

Confirmed:

- Depth-only is not just decorative. On the held-out test split, it has the best baseline accuracy, weighted F1, ECE, and fewer high-confidence errors than scan-only.
- Scan-only has the best test macro F1 among the baseline models, meaning it is slightly better balanced across minority classes.
- Simple scan+depth concatenation does not reliably improve performance. It underperforms depth-only on both val and test.
- Class-weighted focal loss reduces high-confidence errors, especially on validation, but it lowers overall accuracy and weighted F1.

Practical reading:

- Depth is useful and should stay in the project.
- Fusion should not be claimed as solved yet.
- The next fusion version should be mechanism-aware, for example a gated model that trusts depth more under perception degradation or close-obstacle situations.

## Class Behavior

Depth-only test behavior:

| action | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| NORTH | 57 | 0.7538 | 0.8596 | 0.8033 |
| SOUTH | 573 | 0.9210 | 0.8743 | 0.8970 |
| EAST | 144 | 0.6087 | 0.7778 | 0.6829 |
| WEST | 359 | 0.9765 | 0.9248 | 0.9499 |

The weakest class remains `EAST`, mostly because it is a minority class and is confused with `SOUTH` under disturbance.

## Residual Error Mechanisms

High-confidence error patterns:

| modality/model | split | scenario | actual -> predicted | count |
|---|---|---|---|---:|
| scan baseline | test | perception_degradation | SOUTH -> EAST | 43 |
| depth baseline | test | perception_degradation | SOUTH -> EAST | 24 |
| fusion baseline | test | perception_degradation | SOUTH -> EAST | 25 |
| scan focal | test | perception_degradation | SOUTH -> EAST | 32 |
| fusion focal | test | perception_degradation | SOUTH -> EAST | 30 |
| depth baseline | test | external_path_blockage | EAST -> SOUTH | 10 |
| scan baseline | test | external_path_blockage | EAST -> SOUTH | 9 |
| depth baseline | test | external_path_blockage | WEST -> SOUTH | 9 |

Mechanism-level interpretation:

- The dominant residual mechanism is still perception-degradation directional confusion, especially `SOUTH -> EAST`.
- Depth reduces this specific high-confidence error compared with scan-only, but does not eliminate it.
- Fusion does not yet suppress the residual mechanism; it appears to inherit errors from both streams.

## Limitations

- This is still simulation-derived evidence, not real-robot validation.
- There is one seed per split, so the result is useful but not final multi-seed statistical evidence.
- `STAY` remains a recovery/safety-supervisor action, not a Nav2-plan movement-policy label.
- The current fusion is simple feature-level concatenation. It is not yet an uncertainty-gated or mechanism-aware fusion model.

## Next Step

Build the recovery-routing analysis from residual policy evidence:

1. Use depth-only as the current best baseline for test reliability.
2. Treat focal models as uncertainty/error-reduction candidates, not default upgrades.
3. Build a mechanism table for high-confidence `SOUTH -> EAST` under perception degradation.
4. Prototype a gated recovery route:
   - if perception degradation is high and model predicts an east/south directional conflict with high confidence, route to cautious mode or replan;
   - if path blockage causes east/west-to-south confusion, route to replan;
   - keep relocalization separate for future localization-drift data.
