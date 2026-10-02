#!/bin/bash
# RoBERTa-Large on four GLUE tasks, three seeds each.
# One run takes about 50 minutes on a single GPU.

for TASK in mnli qnli sst2 qqp; do
    for SEED in 40 41 42; do
        python main_glue.py --task ${TASK} --alpha 0.5 --seed ${SEED}
    done
done

python summarize.py
