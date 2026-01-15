"""
This file is based on the original implementation of:
BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain
(Gu et al., 2017)

Original implementation:
https://github.com/verazuo/badnets-pytorch

The original code supports training and evaluating a backdoored model
with fixed trigger placement and a single experimental configuration.

==================== Project Extensions ====================

The following extensions were added as part of this final project:

1. Experiment Reproducibility and Logging
   - Added a run_name argument to uniquely identify each experiment run.
   - Modified checkpoint and CSV log filenames to prevent overwriting results
     across different experiments.

2. Trigger Position Control
   - Added support for configurable trigger placement via the --trigger_pos
     argument (br, bl, tr, tl, center).
   - Enables systematic analysis of trigger location sensitivity.

3. Structured Experiment Support
   - Enabled external experiment scripts (experiments/exp02, exp03, exp04)
     to control main.py via CLI arguments.
   - Allows controlled parameter sweeps (poisoning rate, trigger size,
     trigger position).

4. Extended Metadata Logging
   - Each training log CSV now records experimental parameters
     (poisoning_rate, trigger_size, trigger_pos, optimizer, learning rate, etc.)
     to support post-hoc analysis and visualization.

5. Reproducibility via Fixed Random Seed (Added)
   - Added a --seed argument and deterministic settings to make results
     reproducible across runs on the same environment.

============================================================

All core model architecture, dataset poisoning logic, and training procedures
remain faithful to the original implementation.
Responsibility for the original algorithm lies with the original authors.
"""


import argparse
import os
import pathlib
import re
import time
import datetime
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch
from models import BadNet


parser = argparse.ArgumentParser(
    description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".'
)
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='if set: load trained local model to evaluate (no training)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Num workers for DataLoader, default: 0')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate, default: 0.01')
parser.add_argument('--download', action='store_true', help='if set: download data')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./data/)')
parser.add_argument('--device', default='cpu', help='device: cpu | cuda | cuda:0 | mps (default: cpu)')

# Project extension: reproducibility
# Adds a fixed random seed so that runs with the same configuration
# produce similar results across executions.

parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')

# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, 0..1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='target label for backdoor (int, default: 1)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

# Project extension: trigger position control
# The original implementation uses a fixed trigger location.
# This option allows placing the trigger at different positions
# (corners or center) to study the effect of trigger location.

parser.add_argument(
    '--trigger_pos',
    default='br',
    help='Trigger position: br|bl|tr|tl|center (default: br)'
)

# Project extension: run naming
# Adds a short name to each run and includes it in log and checkpoint filenames.
# This helps avoid overwriting results and keeps experiments organized.

parser.add_argument(
    '--run_name',
    default='default',
    help='A short tag to identify this run (e.g., exp02_pr0p03). Used in log/checkpoint filenames.'
)

args = parser.parse_args()


# Project extension: fix random seeds to make experiment results repeatable
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism settings (mostly relevant for CUDA; harmless on CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(device_str: str) -> torch.device:
    """
    Choose torch.device based on user input.
    Supports: cpu, cuda, cuda:0, mps
    """
    if re.match(r'^cuda:\d+$', device_str):
        cuda_num = device_str.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device_str == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device_str == 'mps':
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    return torch.device('cpu')


# Project extension: sanitize strings so they can be safely used in filenames
def safe_tag(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'[^a-zA-Z0-9_\-\.]+', '', s)
    return s if s else "default"

# Project extension: build a unique filename suffix based on the experiment settings
def build_run_suffix() -> str:
    pr_str = str(args.poisoning_rate).replace('.', 'p')
    rn = safe_tag(args.run_name)
    pos = safe_tag(args.trigger_pos)

    suffix = (
        f"{args.dataset}_trigger{args.trigger_label}"
        f"_pr{pr_str}_ts{args.trigger_size}"
        f"_pos{pos}"
        f"_ep{args.epochs}_{args.optimizer}_lr{args.lr}_{args.loss}"
        f"_seed{args.seed}"
        f"__{rn}"
    )
    return suffix


def main():
    print("{}".format(args).replace(', ', ',\n'))

    # Project extension: set seed before any dataset or model is created
    set_seed(args.seed)
    print(f"\n# Seed fixed to: {args.seed}")

    device = pick_device(args.device)
    print(f"\n# Using device: {device}")

    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)

    # Project extension: make DataLoader shuffling deterministic
    g = torch.Generator()
    g.manual_seed(args.seed)

    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g
    )
    data_loader_val_clean = DataLoader(
        dataset_val_clean, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g
    )
    data_loader_val_poisoned = DataLoader(
        dataset_val_poisoned, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=g
    )

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    run_suffix = build_run_suffix()

    model_path = f"./checkpoints/badnet-{run_suffix}.pth"
    log_csv_path = f"./logs/{run_suffix}.csv"

    start_time = time.time()

    if args.load_local:
        print("## Load model from : %s" % model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Tip: run training first, or specify the same --run_name/params used to create the checkpoint."
            )
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")

    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []

        for epoch in range(args.epochs):
            train_stats = train_one_epoch(
                data_loader_train, model, criterion, optimizer, args.loss, device
            )
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)

            print(
                f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} "
                f"Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n"
            )

            torch.save(model.state_dict(), model_path)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "dataset": args.dataset,
                "trigger_label": args.trigger_label,
                "poisoning_rate": args.poisoning_rate,
                "trigger_size": args.trigger_size,
                "trigger_pos": args.trigger_pos,
                "seed": args.seed,
                "optimizer": args.optimizer,
                "lr": args.lr,
                "loss_type": args.loss,
                "run_name": args.run_name,
                "checkpoint_path": model_path,
            }

            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv(log_csv_path, index=False, encoding='utf-8')

        print(f"# Saved training log CSV to: {log_csv_path}")
        print(f"# Saved checkpoint to: {model_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
