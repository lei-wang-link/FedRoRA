"""
data.py — GLUE data partition and shared classifier head.

Call init_shared_cls() once at program start. Every client model is then given
the same, frozen classification head through apply_shared_cls(model).
"""

import gc
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification

import config

# Shared classifier state — populated once by init_shared_cls().
SHARED_CLS_STATE: dict = {}


def init_shared_cls(num_labels: int):
    """Draw the classification head shared (and kept frozen) by all clients."""
    torch.manual_seed(config.DATA_SEED)
    ref = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=num_labels,
        cache_dir=config.MODEL_CACHE, ignore_mismatched_sizes=True)
    state = {n: p.data.detach().clone().cpu()
             for n, p in ref.named_parameters() if "classifier" in n}
    del ref
    gc.collect()
    SHARED_CLS_STATE.clear()
    SHARED_CLS_STATE.update(state)
    print(f"  Shared classifier initialised: {list(SHARED_CLS_STATE.keys())}")


def apply_shared_cls(model):
    """Copy the shared classifier into model and freeze it."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "classifier" not in n:
                continue
            p.data.copy_(SHARED_CLS_STATE[n].to(p.device, dtype=p.dtype))
            p.requires_grad = False


def load_glue_noniid(tokenizer, task: str, alpha: float = 0.5):
    """Dirichlet(alpha) non-IID partition of one GLUE task over all clients.

    Each client gets exactly TRAIN_SAMPLES train and VAL_SAMPLES val examples.
    Dirichlet(alpha) controls the per-client label proportions, not total counts.
    The partition depends only on config.DATA_SEED, so it is identical across
    training seeds.

    Returns {client_id: {"train": Dataset, "validation": Dataset}}.
    """
    np.random.seed(config.DATA_SEED)
    random.seed(config.DATA_SEED)

    client_ids = list(range(config.NUM_CLIENTS))
    col1, col2 = config.TASK_TEXT_KEYS[task]
    num_labels = config.TASK_NUM_LABELS[task]
    raw        = load_dataset("nyu-mll/glue", task)

    def preprocess(examples):
        enc = (tokenizer(examples[col1], padding=False,
                         max_length=128, truncation=True)
               if col2 is None else
               tokenizer(examples[col1], examples[col2], padding=False,
                         max_length=128, truncation=True))
        enc["labels"] = examples["label"]
        return enc

    val_split = config.TASK_VAL_SPLIT.get(task, "validation")
    train_tok = raw["train"].map(preprocess, batched=True,
                                 remove_columns=raw["train"].column_names)
    val_tok   = raw[val_split].map(preprocess, batched=True,
                                   remove_columns=raw[val_split].column_names)

    train_labels = np.array(train_tok["labels"])
    val_labels   = np.array(val_tok["labels"])

    # Per-class index pools, shuffled once
    train_cls_idx = [np.where(train_labels == k)[0].copy() for k in range(num_labels)]
    val_cls_idx   = [np.where(val_labels   == k)[0].copy() for k in range(num_labels)]
    for k in range(num_labels):
        np.random.shuffle(train_cls_idx[k])
        np.random.shuffle(val_cls_idx[k])

    # Draw Dirichlet proportions for each client upfront
    props = {cid: np.random.dirichlet(np.repeat(alpha, num_labels))
             for cid in client_ids}

    used_train: set = set()
    used_val:   set = set()
    client_datasets: dict = {}

    for cid in client_ids:
        p = props[cid]

        # Fixed-size splits: proportions control label skew, not total count
        tr_counts = np.round(p * config.TRAIN_SAMPLES).astype(int)
        va_counts = np.round(p * config.VAL_SAMPLES).astype(int)
        for k in range(num_labels):
            tr_counts[k] = max(tr_counts[k], 1)
            va_counts[k] = max(va_counts[k], 1)
        # Fix rounding residual on the dominant class
        dominant = int(np.argmax(p))
        tr_counts[dominant] += config.TRAIN_SAMPLES - int(tr_counts.sum())
        va_counts[dominant] += config.VAL_SAMPLES   - int(va_counts.sum())
        tr_counts[dominant] = max(tr_counts[dominant], 1)
        va_counts[dominant] = max(va_counts[dominant], 1)

        tr, va = [], []
        for k in range(num_labels):
            # Unused examples first; sample with replacement once a class pool runs out
            avail_tr = [i for i in train_cls_idx[k] if i not in used_train]
            if len(avail_tr) >= tr_counts[k]:
                sel_tr = avail_tr[:tr_counts[k]]
            else:
                sel_tr = avail_tr.copy()
                extra  = np.random.choice(train_cls_idx[k], tr_counts[k] - len(avail_tr),
                                          replace=True).tolist()
                sel_tr.extend(extra)
            tr.extend(sel_tr)
            used_train.update(sel_tr)

            avail_va = [i for i in val_cls_idx[k] if i not in used_val]
            if len(avail_va) >= va_counts[k]:
                sel_va = avail_va[:va_counts[k]]
            else:
                sel_va = avail_va.copy()
                extra  = np.random.choice(val_cls_idx[k], va_counts[k] - len(avail_va),
                                          replace=True).tolist()
                sel_va.extend(extra)
            va.extend(sel_va)
            used_val.update(sel_va)

        random.shuffle(tr)
        random.shuffle(va)

        client_datasets[cid] = {"train": train_tok.select(tr),
                                "validation": val_tok.select(va)}

        tr_dist = {k: int(tr_counts[k]) for k in range(num_labels)}
        va_dist = {k: int(va_counts[k]) for k in range(num_labels)}
        print(f"  Client {cid:2d}  rank={config.RANK_LIST[cid]:2d}"
              f"  train={len(tr)} {tr_dist}  val={len(va)} {va_dist}")

    return client_datasets
