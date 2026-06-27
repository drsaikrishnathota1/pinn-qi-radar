# Strong RunPod Experiment Plan

This package adds the stronger research pipeline.

## Local quick verification

```bash
cd ~/Downloads/pinn-qi-radar
source .venv/bin/activate
pip install -e .
pytest -q
python3 scripts/run_full_experiment.py --quick
```

## Strong RunPod command

```bash
python3 scripts/run_full_experiment.py \
  --n-samples 200000 \
  --pinn-epochs 100 \
  --optimizer-samples 20000 \
  --policy-samples 100000 \
  --policy-epochs 80 \
  --batch-size 2048 \
  --device auto
```

## Main outputs

- `tables/baseline_bounds_summary.csv`
- `tables/pinn_prediction_metrics.csv`
- `tables/qi_optimizer_summary.csv`
- `tables/qi_policy_metrics.csv`
- `figures/fig_pinn_training_loss.png`
- `figures/fig_qi_optimizer_comparison.png`
- `figures/fig_optimizer_runtime.png`
- `figures/fig_neural_policy_gain.png`

## Safe claim

This code supports a simulation-level claim:

A physics-informed neural surrogate can learn plasma-sheath radar-channel degradation, and an AI-based QI policy can select resource-constrained quantum illumination operating parameters that reduce simulated detection-error bounds compared with fixed QI baselines.

Do not claim real hardware detection or near-zero real-world false alarm rates.
