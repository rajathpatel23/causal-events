#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

BATCH=128
LR=1e-3
EPOCHS=

export PYTHONPATH=/home/jovyan/work/causal-events/
export CUDA_VISIBLE_DEVICES=0

python train_classifier.py \
	--model_pretrained_checkpoint="roberta-base" \
    --do_train \
	--dataset_name=abt-buy \
    --train_file /home/jovyan/work/causal-events/data/subtask1/train_subtask1.csv \
	--valid_file /home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=False \
    --output_dir /home/jovyan/work/causal-events/src/report/classification/causal-news-$BATCH-$LR-$EPOCHS-roberta-base/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=5e-05 \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=loss \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG 