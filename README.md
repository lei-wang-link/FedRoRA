# Breaking the Structural Identity: Personalized Federated LoRA Fine-tuning under Rank Heterogeneity

**Accepted by Findings of EMNLP 2026**

**Authors:** Lei Wang*, Jieming Bian*, Letian Zhang, Jie Xu  
(*Equal contribution)

---

## Overview

FedRoRA (Federated Rank-wise Personalized LoRA) is a federated LoRA fine-tuning framework that enables fine-grained personalization for clients with heterogeneous LoRA ranks and non-IID data.

### Key Components

1. **Local Decoupled Parameterization**: Each client decomposes its LoRA update into unit-norm adaptation directions and a learnable diagonal scale of rank-wise magnitudes.
2. **Global Personalized Aggregation**: The server extracts a shared global subspace via truncated SVD, projects each client's update onto it, and returns a personalized initialization through rank-adaptive top-k selection.

## Installation

```bash
git clone https://github.com/lei-wang-link/FedRoRA.git
cd FedRoRA
pip install -r requirements.txt
```

---

## Quick Start

```bash
python main_glue.py --task mnli --alpha 0.5 --seed 42
```

Run all four GLUE tasks with three seeds:

```bash
bash run_glue.sh
```

---

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | mnli | GLUE task (`mnli`, `qnli`, `sst2`, `qqp`) |
| `--alpha` | 0.5 | Dirichlet concentration of the non-IID partition |
| `--seed` | 42 | Random seed |
| `--rounds` | 40 | Number of federated rounds |
| `--output_dir` | ./output | Output directory |
| `--cache_dir` | None | Model cache directory |

Other hyperparameters (client ranks, learning rates, batch size, etc.) are set in `config.py`.

---

## Project Structure

```
FedRoRA/
├── main_glue.py         # Main entry point with CLI
├── fedrora.py           # LoRA layer, client training, server aggregation
├── data.py              # Non-IID data partitioning
├── config.py            # Hyperparameter configuration
├── summarize.py         # Average results over seeds
├── run_glue.sh          # Example run script
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
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
