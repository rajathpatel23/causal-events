import transformers
import torch
from src.model import ContrastivePretrainModel
import torch
import pandas as pd
from src.data.datasets import ContrastivePretrainDataset
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.file_utils import is_offline_mode
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import (
    HfArgumentParser, 
    Trainer,
    TrainingArguments,
    set_seed
)
from dataclasses import dataclass, field
from typing import Optional
import logging
import sys
import os
from src.data.data_collators import DataCollatorContrastivePretrainCausalNews


@dataclass
class ModelArguments:
    model_pretrained_checkpoint:Optional[str] = field()
    do_param_opt:Optional[bool] = field()
    grad_checkpoint:Optional[bool] = field()
    temperature:Optional[float] = field()
    tokenizer:Optional[str] = field()


@dataclass
class DataTrainingArguments:
    train_file:Optional[str] = field()
    valid_file:Optional[str] = field()
    augment:Optional[str] = field()
    train_size:Optional[str] = field()
    max_train_samples:Optional[str] = field()
    max_valid_samples:Optional[str] = field()




if __name__ == '__main__':
    train_path = "/home/jovyan/work/causal-events/data/subtask1/train_subtask1.csv"
    valid_path = "/home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv"

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    train_dataset = ContrastivePretrainDataset(path=train_path, tokenizer='roberta-base', max_length=128)
    valid_dataset = ContrastivePretrainDataset(path=valid_path, tokenizer='roberta-base', max_length=128)
    model = ContrastivePretrainModel(len_tokenizer=128, model='roberta-base', pool='True', proj='mlp', temperature=0.07)

        # Setup logging
    logger = logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    raw_datasets = data_files

    data_collators = DataCollatorContrastivePretrainCausalNews(tokenizer=train_dataset.tokenizer)
    if model_args.model_pretrained_checkpoint:
        model = ContrastivePretrainModel(model_args.model_pretrained_checkpoint, len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, temperature=model_args.temperature)
        if model_args.grad_checkpoint:
            model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)
    else:
        model = ContrastivePretrainModel(len_tokenizer=len(train_dataset.tokenizer), model=model_args.tokenizer, temperature=model_args.temperature)
        if model_args.grad_checkpoint:
            model.encoder.transformer._set_gradient_checkpointing(model.encoder.transformer.encoder, True)

    callback = EarlyStoppingCallback(early_stopping_patience=10)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        data_collator=data_collators
    )
    trainer.args.save_total_limit = 1

        # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        train_dataset.tokenizer.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()




