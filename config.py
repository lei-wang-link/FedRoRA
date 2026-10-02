"""
config.py — Configuration of the GLUE (NLU) experiments.

Values below are the settings used in the paper. main_glue.py fills in the
task-dependent fields (TASK, SEED, GLOBAL_ROUNDS, OUTPUT_ROOT) from the command line.
"""

MODEL_NAME  = "roberta-large"
MODEL_CACHE = None          # local model cache directory; None uses the Hugging Face default
OUTPUT_ROOT = "./output"

# ── Task metadata ─────────────────────────────────────────────────────────────
TASK_TEXT_KEYS = {
    "sst2": ("sentence",  None),
    "qnli": ("question",  "sentence"),
    "qqp":  ("question1", "question2"),
    "mnli": ("premise",   "hypothesis"),
}
TASK_NUM_LABELS = {"sst2": 2, "qnli": 2, "qqp": 2, "mnli": 3}
TASK_VAL_SPLIT  = {"mnli": "validation_matched"}

TASK = "mnli"

# ── Federated setup ───────────────────────────────────────────────────────────
CLIENT_RANKS     = [8, 16, 32, 64]
CLIENTS_PER_RANK = 5
MAX_RANK         = max(CLIENT_RANKS)
NUM_CLIENTS      = len(CLIENT_RANKS) * CLIENTS_PER_RANK      # 20
RANK_LIST        = CLIENT_RANKS * CLIENTS_PER_RANK           # rank of client i

# ── Hyperparameters ───────────────────────────────────────────────────────────
LORA_TARGETS  = ("query", "value")
LORA_DROPOUT  = 0.05
GLOBAL_ROUNDS = 40
LOCAL_EPOCHS  = 2
TRAIN_BATCH   = 128
EVAL_BATCH    = 256
LR            = 5e-4        # learning rate of the directions B, A
LR_S          = 5e-2        # learning rate of the diagonal scale S
TRAIN_SAMPLES = 1000        # training samples per client
VAL_SAMPLES   = 200         # validation samples per client

SEED      = 42              # training seed (--seed)
DATA_SEED = 42              # seed of the data partition and classifier init, fixed across runs
