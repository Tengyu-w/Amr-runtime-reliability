# Gazebo Validation Matrix Results

This is a research-prototype validation summary, not an operational safety claim.

## Medium Matrix Smoke

Run:

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal,external_path_blockage,localization_drift,perception_degradation,execution_deviation,progress_blockage \
  --seeds 10,18 \
  --timeout-sec 34 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_validation_matrix_medium
```

Episode status:

- 12 / 12 episodes produced CSV logs.
- Every episode reached Nav2 navigation start once.
- Goal rejection count was 0 for every episode.
- Final routed mechanisms matched the intended scenario family:
  - `nominal -> NORMAL_NAVIGATION / nominal`
  - `external_path_blockage -> REPLAN / path_blocked`
  - `localization_drift -> RELOCALIZE / localization_uncertainty`
  - `perception_degradation -> HUMAN_REVIEW / perception_degraded`
  - `execution_deviation -> REPLAN / trajectory_deviation`
  - `progress_blockage -> REPLAN / path_blocked`

## Route Model Evidence

Dataset:

- Train seed: 10
- Test seed: 18
- Train rows: 347
- Test rows: 349

Model:

- Two-layer MLP route classifier.
- Inputs are runtime features only.
- Scenario id, fault origin, fault family, and OOD labels are excluded from model inputs.

Headline results:

- Full feature model test accuracy: 0.960
- Full feature model test macro-F1: 0.953
- Risk-only test accuracy: 0.653
- Risk-only test macro-F1: 0.633

This supports the claim that mechanism-relevant runtime features provide route evidence beyond a scalar risk score.

## Fault-Origin Evidence

Expected recovery capture on test seed 18:

- `external_disturbance -> REPLAN`: 0.974
- `state_estimation_drift -> RELOCALIZE`: 0.983
- `perception_degradation -> HUMAN_REVIEW`: 0.950
- `execution_error -> REPLAN`: 0.932
- `none -> NORMAL_NAVIGATION`: 0.947

## Limitations

- This is still a medium smoke matrix: 2 seeds, 6 scenarios.
- The full next validation should use at least 7 seeds across all scenarios.
- Current labels are scenario-level expected routes; later work should add outcome labels such as recovery success, delay-to-recover, collision-free completion, and intervention count.
- Gazebo disturbances are controlled synthetic disturbances. They are useful for mechanism evidence, but not a real-world deployment validation.

## Next Validation

Run the full 6-scenario, 7-seed matrix:

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal,external_path_blockage,localization_drift,perception_degradation,execution_deviation,progress_blockage \
  --seeds 10,11,12,16,17,18,19 \
  --timeout-sec 45 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_validation_matrix
```

## Full Matrix Result

Run:

```bash
python AMR-Runtime-Reliability-Demo/experiments/run_gazebo_validation_matrix.py \
  --scenarios nominal,external_path_blockage,localization_drift,perception_degradation,execution_deviation,progress_blockage \
  --seeds 10,11,12,16,17,18,19 \
  --timeout-sec 80 \
  --retries 1 \
  --out-dir AMR-Runtime-Reliability-Demo/outputs/gazebo_validation_matrix_full_v5
```

Quality gate:

- 42 / 42 episodes completed with `status=ok`.
- Every episode spawned the Gazebo AMR entity.
- Every episode entered Nav2 navigation.
- Goal rejection count was 0 for every accepted episode.
- Split by episode seed: 18 train, 12 validation, 12 test episodes.

Dataset:

- Train rows: 4340
- Validation rows: 2790
- Test rows: 2733

The dataset uses row-wise target labels: nominal rows are labelled `NORMAL_NAVIGATION`; fault-window rows are labelled with the scenario's expected recovery action. This avoids mislabelling temporary blockage recovery tails as persistent `REPLAN`.

Route model:

- Full feature model test accuracy: 0.9996
- Full feature model test macro-F1: 0.9995
- Validation accuracy: 0.9996
- Train accuracy: 1.0000

Ablation:

- Full feature test accuracy: 0.9989
- Risk-only test accuracy: 0.9634
- Risk-only test macro-F1: 0.9516

Fault-window capture on the test split:

- `external_disturbance / REPLAN`: 1.000 for blockage rows.
- `state_estimation_drift / RELOCALIZE`: 1.000 for localization-drift rows.
- `perception_degradation / HUMAN_REVIEW`: 1.000 for perception-degraded rows.
- `execution_error / REPLAN`: 1.000 for execution-deviation rows.
- `none / NORMAL_NAVIGATION`: 1.000 for nominal rows.

Residual limitations:

- The route labels are still generated from controlled scenario semantics and runtime mechanism labels, not from real-world intervention outcomes.
- Full-feature and risk-only results are both high because the synthetic faults are deliberately separable; harder mixed and boundary cases should be added next.
- Next metrics should include recovery latency, path completion, collision-free completion, and intervention counts.
