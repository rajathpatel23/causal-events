#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

BATCH=16
LR=2e-5
EPOCHS=8
MAX_LEN=256
DATE=2023-08-23
FRAC=1-0
METRIC=eval_f1

export PYTHONPATH=/home/jovyan/work/causal-events/
export CUDA_VISIBLE_DEVICES=0

python train_classifier.py \
    --do_train \
	--dataset_name="causal-news" \
    --train_file /home/jovyan/work/causal-events/data/subtask1/train_subtask1.csv \
	--valid_file /home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv \
	--test_file  /home/jovyan/work/causal-events/data/subtask1/test_subtask1_text.csv \
	--do_eval \
	--do_predict \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir /home/jovyan/work/causal-events/src/report/classification/date-08-23-2023/causal-news-$MAX_LEN-$BATCH-$LR-$EPOCHS-$DATE-NO-PRETRAINING-FROZEN-$FRAC-$METRIC-roberta-base/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model="eval_f1" \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	--max_length=$MAX_LEN \
	--dataset_fraction=1.0