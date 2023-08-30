BATCH=256
LR=5e-5
TEMP=0.07
EPOCHS=5
AUG="del"
MAX_LEN=128
SIMCLR=False
DATE=2023-08-25-only-train

export PYTHONPATH=/home/jovyan/work/causal-events
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --do_train \
    --train_file /home/jovyan/work/causal-events/data/subtask1/train_subtask1.csv \
    --valid_file /home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv \
	--tokenizer="roberta-base" \
	--grad_checkpoint=True \
    --output_dir /home/jovyan/work/causal-events/src/report/contrastive/causal-news-$MAX_LEN-$BATCH-$LR-$TEMP-$EPOCHS-$DATE-$SIMCLR-roberta-base/ \
	--temperature=$TEMP \
	--per_device_train_batch_size=$BATCH \
	--learning_rate=$LR \
	--weight_decay=0.01 \
	--num_train_epochs=$EPOCHS \
	--lr_scheduler_type="linear" \
	--warmup_ratio=0.05 \
	--max_grad_norm=1.0 \
	--fp16 \
	--dataloader_num_workers=4 \
	--disable_tqdm=True \
	--save_strategy="epoch" \
	--logging_strategy="epoch" \
	--augment=$AUG \
	--max_length=$MAX_LEN