#!/usr/bin/env python3
import copy
import itertools
from pathlib import Path
import torch
import yaml
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from train import load_config, get_loaders
from models.den import DEN
from trainers.trainer import Trainer

cfg = load_config('configs/split_mnist.yaml')
cfg.setdefault('replay_per_task', 800)
cfg.setdefault('replay_fraction', 0.6)

sweep = []
# Two consolidation strengths, three KD alphas => 6 runs
consolidation_gls = [0.01, 0.001]
kd_alphas = [0.0, 0.5, 1.0]
for c, k in itertools.product(consolidation_gls, kd_alphas):
    sweep.append({'consolidation_gl': c, 'kd_alpha': k})

device = torch.device('cpu')

for i, job in enumerate(sweep):
    run_dir = Path(f"results/sweep_run_{i}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # build model and trainer
    train_loaders, val_loaders, test_loaders, meta = get_loaders(cfg)
    model = DEN(
        input_dim=cfg.get('input_dim',784),
        hidden_dims=cfg.get('hidden_dims',[256,128]),
        num_classes=cfg.get('num_classes',2),
        ex_k=cfg.get('ex_k',10),
        l1_lambda=cfg.get('l1_lambda',1e-5),
        l2_lambda=cfg.get('l2_lambda',1e-4),
        gl_lambda=cfg.get('gl_lambda',0.001),
        regular_lambda=cfg.get('regular_lambda',0.5),
        loss_thr=cfg.get('loss_thr',0.01),
        spl_thr=cfg.get('spl_thr',0.05),
        depth_growth_enabled=cfg.get('depth_growth_enabled', False),
        depth_growth_config=cfg.get('depth_growth_config', {}),
    )
    # apply consolidation strength
    model.consolidation_gl = job['consolidation_gl']
    trainer = Trainer(model, device, log_dir=str(run_dir), resume_from=None)
    # set kd alpha on trainer so trainer will pass it into add_task
    trainer.kd_alpha = job['kd_alpha']
    trainer.replay_per_task = cfg.get('replay_per_task',trainer.replay_per_task)
    model.replay_fraction = cfg.get('replay_fraction', getattr(model, 'replay_fraction', 0.3))

    print(f"Running sweep {i+1}/{len(sweep)}: consolidation_gl={job['consolidation_gl']} kd_alpha={job['kd_alpha']}")
    history = trainer.train(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        max_iter=cfg.get('max_iter',500),
        lr=cfg.get('lr', 0.001),
        batch_size=cfg.get('batch_size',256),
        verbose=True,
    )
    # save the history
    with open(run_dir / 'history.yaml', 'w') as f:
        yaml.safe_dump(history, f)

print('Sweep finished')
