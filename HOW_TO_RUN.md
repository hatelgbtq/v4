# DEN DepthGrowth v2 — laptop quick start

Python 3.14 + PyTorch (CPU) in `.venv`. Data auto-downloads to `data/`
on first use (CIFAR-100; ~160 MB tarball, needs a few minutes).

## Setup

```bash
python -m venv .venv                # Python 3.14
.venv/bin/pip install torch torchvision numpy pyyaml matplotlib scikit-learn
```

## Benchmark (Split CIFAR-100, 8 variants x 3 seeds)

```bash
.venv/bin/python bench_cifar100.py --tasks 5 --max-iter 500 \
    --seeds 1004 1005 1006
```

- Results: `results/bench_cifar100/<variant>_seed<seed>/metrics.json`
- Auto-written: `REPORT.txt` + `summary.json` in `results/bench_cifar100/`
- Cached runs are skipped on re-run (cache key = variant + seed).
- Single quick run: `--variants v2_val_loss_plateau --seeds 1004`
- Force re-run: `--force`

### Progress check

```bash
.venv/bin/python bench_status.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # 25 passed
```

## Train one config (verbose)

```bash
.venv/bin/python train.py --config configs/split_cifar100.yaml \
    --max-iter 500 --log-dir results/split_cifar100 --no-plots
```

## Config keys (calibrated for CIFAR-100)

- `saturation_ratio: 0.02`, `saturation_threshold: 0.3`
- `imbalance_ratio: 0.7`
- `cka_threshold: 0.7`
- `patience: 1`

NOTE: YAML `1e-05` parses as a *string* in PyYAML — write decimals
(`0.00001`) in configs.