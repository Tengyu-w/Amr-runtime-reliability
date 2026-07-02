# Neural Policy Evidence Layer

This is the direct model layer for the AMR simulation project.

Instead of predicting only whether recovery is needed, the policy model predicts
the recovery action itself:

```text
NORMAL_NAVIGATION
CAUTIOUS_MODE
REPLAN
RELOCALIZE
HUMAN_REVIEW
SAFE_STOP
```

## Model

`experiments/train_policy_model.py` trains a PyTorch MLP policy:

```text
observable state features
  -> 32-unit hidden layer
  -> 32-unit hidden layer
  -> action logits
```

The model is trained by imitation from `mechanism_router` actions in the
simulation dataset. This is not yet full RL, but it gives the evidence we need
for ECG-style analysis:

- action probabilities;
- policy entropy;
- top-action margin;
- action logits;
- hidden embedding distances to action centroids;
- source-specific model errors.

The model input does not include fault labels, scenario labels, or OOD labels.
Those are used only after inference to evaluate whether the model evidence
aligns with the true simulated fault source.

## Run

```powershell
python .\experiments\train_policy_model.py --dataset-dir .\outputs\scenario_dataset_smoke --out-dir .\outputs\policy_model_smoke
```

Outputs:

- `policy_scores.csv`
- `policy_metrics.csv`
- `policy_report.json`

## Next Analysis

The next step is to analyze where the policy is uncertain or wrong:

- high entropy by fault source;
- low margin around action boundaries;
- embedding overlap between `CAUTIOUS_MODE` and `REPLAN`;
- whether OOD-style goal shifts form a separate region;
- whether confident wrong actions remain after mechanism-aware supervision.
