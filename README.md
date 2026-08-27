# Dynamically Expandable Networks (DEN)

**PyTorch reimplementation of *"Lifelong Learning with Dynamically Expandable Networks"* (Yoon et al., ICLR 2018)**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

DEN is a continual-learning architecture that **dynamically expands** its capacity as new tasks arrive, learns a **compact overlapping knowledge-sharing structure**, and **preserves previously learned knowledge** through timestamped neuron splitting and selective retraining.

### Key Ideas

| Component | Purpose |
|-----------|---------|
| **Selective Retraining** | Identify and train only the sub-network relevant to the new task, freezing shared parameters |
| **Dynamic Expansion** | Add new hidden units when the current capacity is insufficient (measured by loss threshold) |
| **Group Sparsity** | Apply group-lasso regularisation on newly added units to prune redundant ones |
| **Split & Duplication** | Detect drifted neurons and duplicate them — one copy preserves old knowledge, one adapts |
| **Timestamping** | Tag each neuron with the task that created it; during inference, only activate neurons from relevant tasks |
| **Knowledge Preservation** | L2 regularisation toward previous task weights to prevent catastrophic forgetting |

### Architecture Diagram

```
                    ┌──────────────┐
                    │   Task 1     │
                    │   Output     │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │  Layer 2   │  Layer 2   │ ← neurons timestamped per task
              │  (task 1)  │  (task 2)  │
              └────────────┼────────────┘
                           │
              ┌────────────┼────────────┐
              │  Layer 1   │  Layer 1   │ ← split & expanded units
              │  (task 1)  │  (task 2)  │
              └────────────┼────────────┘
                           │
                      ┌────┴────┐
                      │  Input  │
                      └─────────┘
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DEN.git
cd DEN

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Permuted MNIST (10 tasks)

```bash
python train.py --config configs/permuted_mnist.yaml
```

### Split MNIST (5 tasks, 2 classes each)

```bash
python train.py --config configs/split_mnist.yaml
```

### Standard MNIST (single-task baseline)

```bash
python train.py --config configs/mnist.yaml
```

### Custom configuration via CLI

```bash
python train.py \
    --dataset permuted_mnist \
    --num-tasks 5 \
    --max-iter 2000 \
    --lr 0.0005 \
    --ex-k 15 \
    --log-dir results/my_experiment
```

---

## Configuration

All hyper-parameters are specified in YAML files under `configs/`.

```yaml
# configs/permuted_mnist.yaml
dataset: permuted_mnist
num_tasks: 10
batch_size: 256
max_iter: 5000
lr: 0.001

# Network architecture
input_dim: 784
hidden_dims: [312, 128]
num_classes: 10

# DEN hyper-parameters
ex_k: 10                    # Neurons added per expansion / split
l1_lambda: 0.00001          # L1 sparsity (soft-thresholding)
l2_lambda: 0.0001           # L2 weight decay
gl_lambda: 0.001            # Group-lasso on new units
regular_lambda: 0.5         # Knowledge preservation strength
loss_thr: 0.01              # Loss threshold for triggering expansion
spl_thr: 0.05               # Drift threshold for triggering splitting

# Training
device: auto                # 'auto', 'cpu', or 'cuda:N'
log_dir: results/permuted_mnist
seed: 1004
```

---

## Project Structure

```
DEN/
├── datasets/
│   ├── mnist.py            # Standard MNIST loader
│   ├── permuted_mnist.py   # Permuted MNIST (multi-task benchmark)
│   └── split_mnist.py      # Split MNIST (5 binary tasks)
├── models/
│   ├── den.py              # Main DEN model (lifecycle, training phases)
│   ├── layers.py           # DynamicLinear, TaskOutputHead
│   ├── grow.py             # Network expansion + group-lasso utilities
│   ├── prune.py            # Selective retraining + neuron selection
│   └── utils.py            # Metrics, knowledge-preservation loss
├── trainers/
│   └── trainer.py          # Multi-task orchestration + evaluation
├── configs/
│   ├── mnist.yaml
│   ├── permuted_mnist.yaml
│   └── split_mnist.yaml
├── tests/
│   ├── test_layers.py              # Expansion, splitting, timestamp tests
│   ├── test_grow.py                # Group-lasso, pruning tests
│   ├── test_timestamp.py           # Timestamp masking tests
│   └── test_selective_retrain.py   # Neuron selection tests
├── visualization/
│   └── visualize.py        # Accuracy, forgetting, growth plots
├── train.py                # Entry point
├── requirements.txt
└── README.md
```

---

## Algorithm Details

### Phase 1: Selective Retraining

1. Freeze all shared (hidden) layers.
2. Train only the task-specific output head for a small number of iterations.
3. Identify **active neurons** by walking backward from non-zero output weights.
4. Construct a sub-network from active neurons only.
5. Train the sub-network with knowledge-preservation L2 regularisation.
6. Merge trained sub-weights back into the full network.

### Phase 2: Dynamic Expansion

If the loss after selective retraining exceeds `loss_thr`:

1. Add `ex_k` new hidden units to each layer.
   - First layer: append output columns only.
   - Intermediate layers: append both output columns and input rows.
   - Output layer: append input rows for all task heads.
2. Train with **group-lasso** regularisation on newly added units.
3. Remove (prune) units whose group norm is zero after group-lasso.

### Phase 3: Split & Duplication

1. For each hidden layer, compute per-neuron **drift**:
   `||prev_weight[j] - cur_weight[j]||_2`
2. If drift > `spl_thr`, the neuron is "drifted".
3. Select top `ex_k` drifted neurons (by drift magnitude).
4. For each drifted neuron, create two copies:
   - **Old copy**: preserves the original weights (frozen for previous tasks).
   - **New copy**: receives current weights (trainable for the new task).
5. Train the split network with knowledge preservation.

### Timestamping

Each neuron carries a timestamp indicating during which task it was created. During inference for task *t*, only neurons with timestamp ≤ *t* are activated, preserving the original sub-network structure.

---

## Reproducing Original Experiments

| Dataset | Tasks | Architecture | Expected Performance |
|---------|-------|-------------|---------------------|
| Permuted MNIST | 10 | [784, 312, 128, 10] | ~90% average accuracy |
| Split MNIST | 5 | [784, 256, 128, 2] | ~98% average accuracy |

Run with the provided config files:

```bash
python train.py --config configs/permuted_mnist.yaml
python train.py --config configs/split_mnist.yaml
```

---

## Outputs

After training, the following are saved to the log directory:

- `metrics.json` — all numerical results
- `accuracy.png` — test accuracy per task over time
- `forgetting.png` — average forgetting after each task
- `neuron_growth.png` — neuron count per hidden layer
- `neurons_added_split.png` — neurons added vs. split per task
- `sparsity.png` — parameter sparsity over time
- `parameter_growth.png` — total parameter count over time

### Example Plots

*(Plots are generated automatically during training)*

---

## Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_layers.py -v
```

---

## Extending DEN

The modular design makes it easy to experiment with new ideas:

- **New dataset**: Add a loader in `datasets/` and register it in `train.py`.
- **New growth strategy**: Subclass or modify functions in `models/grow.py`.
- **New pruning strategy**: Modify `models/prune.py`.
- **New regularisation**: Add the loss term in `models/den.py`.
- **CIFAR / Tiny ImageNet**: Add convolutional layers that use `DynamicLinear` internally.

---

## Citation

```bibtex
@inproceedings{yoon2018lifelong,
  title={Lifelong Learning with Dynamically Expandable Networks},
  author={Yoon, Jaehong and Yang, Eunho and Lee, Jeongtae and Hwang, Sung Ju},
  year={2018},
  booktitle={International Conference on Learning Representations (ICLR)}
}
```

---

## License

This project is available for research and educational purposes.
