# Learned Model Evidence Layer

This layer is the bridge from the toy simulator toward the ECG-style workflow.

The simulator provides labelled episodes, but the router should not read
`fault_origin` directly at inference time. Instead, a model is trained from
observable timestep features and audited using hidden evidence:

- `recovery_prob`
- `recovery_entropy`
- `recovery_margin`
- hidden-layer embedding distances to recovery and nominal centroids

## Current Model

`experiments/train_learned_recovery_model.py` trains a small MLP classifier to
predict `needs_hard_recovery`.

Inputs are runtime-observable features such as position, target, risk score,
localization uncertainty, sensor confidence, path blocked score, trajectory
deviation, replanning failures, and progress stagnation.

The model does not use `fault_origin`, `fault_family`, or `scenario_id` as
features. Those labels are used only for training targets and post-hoc
evaluation.

## Why This Matters

This gives us the ECG-style loop:

```text
controlled simulation data
  -> learned model
  -> model probabilities / uncertainty / hidden representation
  -> failure mechanism analysis
  -> mechanism-aware routing
```

The current MLP is only a first smoke model. The next upgrade can replace it
with a learned navigation policy, Q-learning/DQN, or imitation policy, and then
use Q-value margins, policy entropy, value drops, and embedding distances as the
evidence layer.

## Run

```powershell
python .\experiments\train_learned_recovery_model.py --dataset-dir .\outputs\scenario_dataset_smoke --out-dir .\outputs\learned_recovery_smoke
```

Outputs:

- `learned_recovery_scores.csv`
- `learned_recovery_metrics.csv`
- `learned_recovery_report.json`
