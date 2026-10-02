# Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity

**Accepted by EMNLP 2026** &nbsp;|&nbsp; [Paper](https://arxiv.org/abs/2609.00632)

**Authors:** Lei Wang*, Jieming Bian*, Letian Zhang, Jie Xu  
(*Equal contribution)

---

## Overview

FedRoRA (Federated Rank-wise Personalized LoRA) is a federated LoRA fine-tuning framework for clients that differ in both their LoRA rank budgets and their data distributions. Existing rank-heterogeneous methods give every client of the same rank an identical model. FedRoRA instead returns a personalized initialization to each client.

- **Client: decoupled parameterization.** The local update is written as `ΔW_i = B̃_i S_i Ã_i`, where `B̃_i`, `Ã_i` are unit-norm adaptation directions and `S_i` is a learnable diagonal matrix of rank-wise magnitudes.
- **Server: personalized aggregation.** The server averages the client updates, extracts a shared global subspace `(U, V)` by truncated SVD, projects each client's update onto the global rank-one directions (`s_k = u_kᵀ ΔW_i v_k`), and sends client `i` the `r_i` directions it is most aligned with, together with its own coefficients as `S_i`.

This repository contains the FedRoRA implementation and the main NLU experiment of the paper: RoBERTa-Large on GLUE under rank-heterogeneous, non-IID clients.

## Installation

```bash
git clone https://github.com/lei-wang-link/FedRoRA.git
cd FedRoRA
pip install -r requirements.txt
```

The results in the paper were obtained with Python 3.10, PyTorch 2.11.0, Transformers 4.57.0, Datasets 4.8.4 and Accelerate 1.13.0 on a single NVIDIA B200 GPU.

---

## Quick Start

```bash
# 2 communication rounds, a few minutes
python main_glue.py --task sst2 --seed 42 --rounds 2
```

GLUE is downloaded automatically from the Hugging Face Hub.

## Reproducing the Paper Results

```bash
# 4 GLUE tasks x 3 seeds, about 50 minutes per run
bash run_glue.sh
```

or run the tasks one at a time and aggregate afterwards:

```bash
python main_glue.py --task mnli --alpha 0.5 --seed 42
python summarize.py          # mean ± std over the finished seeds
```

For each run, `summarize.py` takes the round with the highest client-averaged accuracy and reports the mean ± std over the seeds 40, 41, 42, which is the protocol used in the paper.

### Experimental Setting

| | |
|---|---|
| Backbone | RoBERTa-Large |
| Tasks | MNLI, QNLI, SST-2, QQP (one task per run) |
| Clients | 20, each with 1,000 training and 200 validation examples |
| LoRA ranks | {8, 16, 32, 64}, 5 clients per rank |
| Data heterogeneity | Dirichlet label skew, α = 0.5 (partition fixed across seeds) |
| LoRA modules | `query`, `value`, dropout 0.05 |
| Classifier | Shared random initialization, frozen |
| Rounds / local epochs | 40 / 2 |
| Batch size | 128 |
| Learning rate | 5e-4 for `B`, `A`; 5e-2 for `S` |

### Expected Results

Accuracy (%), mean ± std over three seeds:

| Method  | MNLI | QNLI | SST-2 | QQP | Average |
|---------|------|------|-------|-----|---------|
| FedRoRA | 87.48 ± 0.12 | 92.30 ± 0.42 | 95.94 ± 0.05 | 90.03 ± 0.12 | 91.44 |

Small deviations are possible with other library versions or GPU models.

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | mnli | GLUE task: `mnli`, `qnli`, `sst2`, `qqp` |
| `--alpha` | 0.5 | Dirichlet concentration of the non-IID label partition |
| `--seed` | 42 | Training seed (the data partition is fixed) |
| `--rounds` | 40 | Number of communication rounds |
| `--output_dir` | ./output | Output directory |
| `--cache_dir` | None | Model cache directory |

The remaining hyperparameters are defined in `config.py`.

### Outputs

Each run writes to `output/glue/<task>/alpha<α>/seed<seed>/`:

- `log.txt`: client-averaged accuracy after every round
- `results.json`: per-client accuracy for every round

---

## Project Structure

```
FedRoRA/
├── main_glue.py         # Entry point
├── fedrora.py           # Decoupled LoRA layer, server aggregation, federated training loop
├── data.py              # GLUE non-IID partition and shared classifier head
├── config.py            # Experiment configuration
├── summarize.py         # Mean ± std over seeds
├── run_glue.sh          # All tasks and seeds
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── README.md            # This file
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{wang2026fedrora,
  title={Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity},
  author={Wang, Lei and Bian, Jieming and Zhang, Letian and Xu, Jie},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
