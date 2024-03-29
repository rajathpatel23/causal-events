#!/bin/bash
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

BATCH=8
LR=2e-5
EPOCHS=8
MAX_LEN=128
DATE=2023-08-25-8-only-train
FRAC=1.0
METRIC=eval_f1

export PYTHONPATH=/home/jovyan/work/causal-events/
export CUDA_VISIBLE_DEVICES=0

python train_classifier.py \
	--model_pretrained_checkpoint /home/jovyan/work/causal-events/src/report/contrastive/causal-news-128-256-5e-5-0.07-5-2023-08-25-only-train-False-roberta-base/pytorch_model.bin \
    --do_train \
	--dataset_name="causal-news" \
    --train_file /home/jovyan/work/causal-events/data/subtask1/train_subtask1.csv \
	--valid_file /home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv \
	--test_file  /home/jovyan/work/causal-events/data/subtask1/test_subtask1_text.csv \
	--do_eval \
	--do_predict \
	--evaluation_strategy=epoch \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /home/jovyan/work/causal-events/src/report/classification/date-08-25-2023-opps/causal-news-$MAX_LEN-$BATCH-$LR-$EPOCHS-$DATE-NOT-FROZEN-$FRAC-$METRIC-roberta-base/ \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--metric_for_best_model=$METRIC \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--load_best_model_at_end \
	--augment=$AUG \
	--max_length=$MAX_LEN \
	--dataset_fraction=$FRAC

#	--model_pretrained_checkpoint /home/jovyan/work/causal-events/src/report/contrastive/causal-news-256-128-5e-5-0.07-5-False-2023-08-23-opps-roberta-base/pytorch_model.bin \

