# Accuracy Enhancements Before RunPod

This package adds:

1. Multi-seed robustness experiments
2. Very-low to extreme plasma stress tests
3. Final manuscript summary and key-claims tables
4. Updated full experiment runner with stress/final table flags

## Local quick test

```bash
cd ~/Downloads/pinn-qi-radar
source .venv/bin/activate
pip install -e .
pytest -q
python3 scripts/run_full_experiment.py --quick
```

## Recommended RunPod main run

```bash
python3 scripts/run_full_experiment.py \
  --n-samples 200000 \
  --pinn-epochs 100 \
  --optimizer-samples 20000 \
  --policy-samples 100000 \
  --policy-epochs 80 \
  --batch-size 2048 \
  --device auto \
  --stress-test \
  --stress-samples 50000 \
  --final-tables
```

## Optional multi-seed RunPod robustness run

```bash
python3 scripts/07_run_multiseed_experiments.py \
  --seeds 42 43 44 45 46 \
  --n-samples 100000 \
  --pinn-epochs 80 \
  --optimizer-samples 10000 \
  --policy-samples 50000 \
  --policy-epochs 60 \
  --batch-size 2048 \
  --device auto
```

## Final paper tables

- `tables/final_manuscript_summary.csv`
- `tables/final_key_claims.csv`
- `tables/multiseed_pinn_metrics_mean_std.csv`
- `tables/multiseed_optimizer_mean_std.csv`
- `tables/multiseed_policy_mean_std.csv`
- `tables/stress_baseline_summary.csv`
- `tables/stress_optimizer_summary.csv`
