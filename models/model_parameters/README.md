# Saved model parameters

One JSON file per finalized model, written by `save_model` in `src/tools/ml_models/registry.py`
(see `docs/ml_plan.md`, *Saved models*). Each file is named by the model's id:

```
{model}_{target}_{ticker}_{last bar}_{hash}.json      e.g. lightgbm_wdl_SPY_20260930_3f9a1c2e.json
```

The hash covers the methodology (model class and settings, features, win/draw/loss threshold,
taus, walk-forward, the model code) and a fingerprint of the bars. The same methodology on the same
bars always gets the same id, and any change gets a new one. `spec_id` inside the file is the
methodology's hash alone, the same for every ticker trained that way.

These files are tracked. They hold settings, the data's date range and fingerprint, the final fit
(trees, epochs), summary scores of the walk-forward forecasts, and library versions. They hold no
per-day values and no trained weights.

The models themselves, with their walk-forward forecasts, are in `models/models/<id>.pkl`. That
folder is git-ignored because the models are trained on market data, so a model is only on the
computer that saved it. `list_models()` reads this folder and shows which models are present
locally (`saved_here`).
