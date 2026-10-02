"""
main_glue.py — FedRoRA on GLUE with RoBERTa-Large.

20 clients share one GLUE task under a Dirichlet(alpha) label-skewed partition,
with heterogeneous LoRA ranks {8, 16, 32, 64} (5 clients per rank).
The data partition is fixed across runs; --seed only changes the training seed.

Usage:
    python main_glue.py --task mnli --seed 42
    python main_glue.py --task sst2 --seed 40 --rounds 2      # quick test
"""

import argparse
import json
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from transformers import AutoTokenizer

import config
import data
import fedrora


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  default="mnli", choices=["mnli", "qnli", "sst2", "qqp"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration of the non-IID label partition.")
    parser.add_argument("--seed",  type=int, default=42,
                        help="Training seed. The data partition is always seeded with 42.")
    parser.add_argument("--rounds", type=int, default=config.GLOBAL_ROUNDS,
                        help="Number of communication rounds.")
    parser.add_argument("--output_dir", default=config.OUTPUT_ROOT)
    parser.add_argument("--cache_dir", default=None,
                        help="Model cache directory (default: Hugging Face cache).")
    args = parser.parse_args()

    config.TASK          = args.task
    config.SEED          = args.seed
    config.GLOBAL_ROUNDS = args.rounds
    config.OUTPUT_ROOT   = args.output_dir
    config.MODEL_CACHE   = args.cache_dir

    output_dir = os.path.join(config.OUTPUT_ROOT, "glue", args.task,
                              f"alpha{args.alpha}", f"seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"FedRoRA  task={args.task}  alpha={args.alpha}  "
          f"seed={args.seed}  data_seed={config.DATA_SEED}")
    print(f"clients={config.NUM_CLIENTS}  ranks={config.CLIENT_RANKS}  "
          f"rounds={config.GLOBAL_ROUNDS}  LR={config.LR}  LR_S={config.LR_S}")
    print(f"output: {output_dir}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME,
                                              cache_dir=config.MODEL_CACHE)

    # ── Shared classifier and data partition: always seeded with DATA_SEED ────
    random.seed(config.DATA_SEED)
    np.random.seed(config.DATA_SEED)
    torch.manual_seed(config.DATA_SEED)

    num_labels = config.TASK_NUM_LABELS[args.task]
    print(f"\nInitialising shared classifier ({num_labels} labels) ...")
    data.init_shared_cls(num_labels)

    print("\nLoading datasets (non-IID Dirichlet) ...")
    client_datasets = data.load_glue_noniid(tokenizer, args.task, alpha=args.alpha)

    # ── Federated training ────────────────────────────────────────────────────
    scores = fedrora.train(client_datasets, tokenizer, output_dir)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in scores.items()}, f, indent=2)

    avgs = [np.mean([scores[i][r] for i in range(config.NUM_CLIENTS)])
            for r in range(config.GLOBAL_ROUNDS)]
    best_round = int(np.argmax(avgs))
    print(f"\nFedRoRA  task={args.task}  seed={args.seed}  "
          f"R{config.GLOBAL_ROUNDS}={avgs[-1]:.4f}  "
          f"best={avgs[best_round]:.4f} (round {best_round+1})")


if __name__ == "__main__":
    main()
