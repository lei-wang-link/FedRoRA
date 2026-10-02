"""
fedrora.py — FedRoRA on GLUE (RoBERTa-Large).

Client      : LoRA with the decoupled parameterisation ΔW_i = B̃_i S_i Ã_i.
              B̃_i, Ã_i are unit-norm directions, S_i a learnable diagonal scale
              trained with its own learning rate.

Server      : average the client updates ΔW_i → truncated SVD (U, Σ, V) →
              project each ΔW_i onto the global rank-one directions u_k v_kᵀ →
              client i receives the r_i directions it is most aligned with,
              together with its own projection coefficients as S_i.

Evaluation  : each client is evaluated on the personalised model it receives
              from the server (no further local training).

Classifier  : shared random init, frozen, never aggregated.
"""

import gc
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification, DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

import config
import data


# ── Decoupled LoRA layer ──────────────────────────────────────────────────────

class BSALinear(nn.Module):
    """LoRA with the decoupled parameterisation ΔW = B̃ S Ã.

    B and A are unit-normalised on-the-fly (columns of B, rows of A), so the
    diagonal scale s carries the magnitude of every rank-one direction.
    """

    def __init__(self, base_linear: nn.Linear, r: int, dropout: float = 0.0):
        super().__init__()
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.r      = r
        self.weight = base_linear.weight          # shared reference, frozen
        self.bias   = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_B = nn.Parameter(torch.empty(out_f, r), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(r, in_f),  requires_grad=False)
        self.lora_s = nn.Parameter(torch.full((r,), 1.0 / math.sqrt(r)),
                                   requires_grad=False)

        # Unit-norm init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data /= self.lora_A.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        self.lora_B.data /= self.lora_B.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def _delta_w(self) -> torch.Tensor:
        B_n = self.lora_B / self.lora_B.norm(dim=0, keepdim=True).clamp(min=1e-8)
        A_n = self.lora_A / self.lora_A.norm(dim=1, keepdim=True).clamp(min=1e-8)
        return (B_n * self.lora_s.unsqueeze(0)) @ A_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.linear(x, self.weight, self.bias) +
                F.linear(self.drop(x), self._delta_w()))

    def get_delta_w(self) -> torch.Tensor:
        with torch.no_grad():
            return self._delta_w()


def apply_bsa(model: nn.Module, r: int, dropout: float) -> nn.Module:
    """Replace every target linear layer of model by a BSALinear of rank r."""
    for key, _ in list(model.named_modules()):
        if not any(key.endswith(t) for t in config.LORA_TARGETS):
            continue
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        child  = model.get_submodule(key)
        if not isinstance(child, nn.Linear):
            continue
        new = BSALinear(child, r, dropout)
        new.to(child.weight.device)
        setattr(parent, key.split(".")[-1], new)
    return model


def get_bsa_layers(model: nn.Module) -> Dict[str, "BSALinear"]:
    return {n: m for n, m in model.named_modules() if isinstance(m, BSALinear)}


# ── Server aggregation ────────────────────────────────────────────────────────

def svd_aggregate(all_dw: List[Dict[str, torch.Tensor]],
                  max_rank: int) -> Dict[str, Tuple]:
    """Average the client updates and extract the global subspace by truncated SVD.

    Clients hold equally sized datasets, so the dataset-weighted average is the mean.
    Returns {layer: (U, Σ, V)} with U, V of max_rank orthonormal columns.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {}
    for key in all_dw[0]:
        avg = torch.stack([dw[key].to(device) for dw in all_dw]).mean(0)
        U, S, Vh = torch.linalg.svd(avg, full_matrices=False)
        R = min(max_rank, S.shape[0])
        result[key] = (U[:, :R].cpu(), S[:R].cpu(), Vh[:R].T.cpu())
    return result


def personalized_topk(all_dw: List[Dict[str, torch.Tensor]],
                      svd_result: Dict[str, Tuple],
                      client_ranks: List[int]) -> List[Dict]:
    """Personalised projection and rank-adaptive top-k selection.

    For client i, s_k = u_kᵀ ΔW_i v_k is its coefficient along the k-th global
    direction. The r_i directions with the largest |s_k| are selected, and their
    signed coefficients become the client's next diagonal scale S_i.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = [{} for _ in range(len(all_dw))]
    for key in svd_result:
        U_full, _, V_full = svd_result[key]
        U_gpu = U_full.to(device)
        V_gpu = V_full.to(device)
        for i, dw in enumerate(all_dw):
            r_i     = client_ranks[i]
            dw_gpu  = dw[key].to(device)
            energy  = ((U_gpu.T @ dw_gpu) * V_gpu.T).sum(dim=1).abs()
            top_idx = energy.topk(r_i).indices.sort().values
            U_i     = U_gpu[:, top_idx]
            V_i     = V_gpu[:, top_idx]
            s_vals  = ((U_i.T @ dw_gpu) * V_i.T).sum(dim=1)
            result[i][key] = {"s": s_vals.cpu(), "idx": top_idx.cpu()}
    return result


# ── Client ────────────────────────────────────────────────────────────────────

class Client:
    def __init__(self, cid, rank, num_labels, dataset, tokenizer):
        self.cid        = cid
        self.rank       = rank
        self.num_labels = num_labels
        self.dataset    = dataset
        self.collator   = DataCollatorWithPadding(tokenizer)
        self.model      = None

    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME, num_labels=self.num_labels,
            cache_dir=config.MODEL_CACHE, ignore_mismatched_sizes=True)
        apply_bsa(self.model, self.rank, config.LORA_DROPOUT)
        data.apply_shared_cls(self.model)

    def unload_model(self):
        del self.model
        self.model = None
        torch.cuda.empty_cache()
        gc.collect()

    def init_from_server(self, svd_result: Dict, selection: Dict):
        """Load the personalised (B̃_i, S_i, Ã_i) sent by the server."""
        for n, m in get_bsa_layers(self.model).items():
            U, _, V = svd_result[n]
            idx = selection[n]["idx"]
            with torch.no_grad():
                m.lora_B.data.copy_(U[:, idx].to(m.lora_B.device))
                m.lora_A.data.copy_(V[:, idx].T.to(m.lora_A.device))
                m.lora_s.data.copy_(selection[n]["s"].to(m.lora_s.device))

    def local_train(self, svd_result: Optional[Dict], selection: Optional[Dict],
                    output_dir: str):
        """Start from the server state (random init in the first round), then train B, S, A."""
        if svd_result is not None:
            self.init_from_server(svd_result, selection)

        for n, p in self.model.named_parameters():
            p.requires_grad = "lora_" in n

        s_params  = [p for n, p in self.model.named_parameters() if "lora_s" in n]
        ab_params = [p for n, p in self.model.named_parameters()
                     if "lora_A" in n or "lora_B" in n]
        optimizer = torch.optim.AdamW([{"params": s_params,  "lr": config.LR_S},
                                       {"params": ab_params, "lr": config.LR}])
        args = TrainingArguments(
            output_dir=os.path.join(output_dir, "trainer"),
            per_device_train_batch_size=config.TRAIN_BATCH,
            num_train_epochs=config.LOCAL_EPOCHS,
            seed=config.SEED,
            fp16=torch.cuda.is_available(),
            save_strategy="no", report_to="none",
            remove_unused_columns=False,
        )
        Trainer(
            model=self.model, args=args,
            train_dataset=self.dataset["train"],
            data_collator=self.collator,
            optimizers=(optimizer, None),
        ).train()

    def get_delta_w(self) -> Dict[str, torch.Tensor]:
        return {n: m.get_delta_w() for n, m in get_bsa_layers(self.model).items()}

    def evaluate(self, output_dir: str) -> float:
        self.model.eval()
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}
        eval_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "trainer"),
            per_device_eval_batch_size=config.EVAL_BATCH,
            fp16=torch.cuda.is_available(), report_to="none",
        )
        metrics = Trainer(
            model=self.model, args=eval_args,
            eval_dataset=self.dataset["validation"],
            data_collator=self.collator,
            compute_metrics=compute_metrics,
        ).evaluate()
        return metrics["eval_accuracy"]


# ── Federated training loop ───────────────────────────────────────────────────

def _seed_initialisation():
    """Set the RNG state from which the first client's round-0 LoRA factors are drawn.

    Everything after that point is governed by --seed, which every local Trainer
    re-applies. The runs reported in the paper were launched from a script that
    evaluated a baseline in the same process right before FedRoRA, which left the
    generator seeded with 42 and advanced by one draw. That state is replayed here
    so that this code reproduces the reported numbers exactly.
    """
    set_seed(42)
    torch.empty((), dtype=torch.int64).random_()


def train(client_datasets, tokenizer, output_dir: str) -> Dict[int, list]:
    """Run FedRoRA. Returns {client_id: [accuracy after round 1, 2, ...]}."""
    os.makedirs(output_dir, exist_ok=True)
    log_path   = os.path.join(output_dir, "log.txt")
    num_labels = config.TASK_NUM_LABELS[config.TASK]

    clients = [
        Client(i, config.RANK_LIST[i], num_labels, client_datasets[i], tokenizer)
        for i in range(config.NUM_CLIENTS)
    ]

    svd_result = None
    selection  = [None] * config.NUM_CLIENTS
    all_scores = {i: [] for i in range(config.NUM_CLIENTS)}

    with open(log_path, "w") as f:
        f.write(f"FedRoRA  task={config.TASK}  ranks={config.CLIENT_RANKS}  "
                f"rounds={config.GLOBAL_ROUNDS}  seed={config.SEED}\n")
        f.write("-" * 60 + "\n")

    _seed_initialisation()

    for rnd in range(config.GLOBAL_ROUNDS):
        print(f"\n{'='*60}\nRound {rnd+1}/{config.GLOBAL_ROUNDS}")

        # ── Local training ────────────────────────────────────────────────────
        all_dw = []
        for c in clients:
            print(f"  [train] Client {c.cid:2d}  rank={c.rank}")
            c.load_model()
            c.local_train(svd_result, selection[c.cid], output_dir)
            all_dw.append(c.get_delta_w())
            c.unload_model()

        # ── Server: global subspace + personalised projection / selection ─────
        svd_result = svd_aggregate(all_dw, config.MAX_RANK)
        selection  = personalized_topk(all_dw, svd_result, config.RANK_LIST)

        # ── Evaluation: each client loads its personalised server model ───────
        for c in clients:
            print(f"  [eval]  Client {c.cid:2d}  rank={c.rank}")
            c.load_model()
            c.init_from_server(svd_result, selection[c.cid])
            all_scores[c.cid].append(c.evaluate(output_dir))
            c.unload_model()

        avg  = np.mean([all_scores[i][-1] for i in range(config.NUM_CLIENTS)])
        line = f"Round {rnd+1:2d}  avg={avg:.4f}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return all_scores
