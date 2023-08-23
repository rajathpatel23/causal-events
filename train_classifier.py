import logging

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import torch
from copy import deepcopy
import json

import transformers
from transformers import (
    HfArgumentParser,
    DataCollatorWithPadding,
    default_data_collator,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed, AutoModelForSequenceClassification
)
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers import EarlyStoppingCallback
from transformers.utils.hp_naming import TrialShortNamer
from src.data.datasets import ContrastiveClassificationDataset, ContrastiveClassificationTestData
from src.data.data_collators import DataCollatorClassification, DataCollatorClassificationTestData
from src.model import ContrastiveClassifierModel
from src.metrics import compute_metrics_bce, compute_metrics_soft_max


check_min_version("4.8.22")
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_pretrained_checkpoint:Optional[str] = field(default=None)
    do_param_opt:Optional[bool] = field(default=False)
    frozen:Optional[bool] = field(default=False)
    grad_checkpoint:Optional[bool] = field(default=True)
    tokenizer:Optional[str] = field(default="roberta-base")
    max_length:Optional[int] = field(default=128)
    dataset_fraction: Optional[float] = field(default=1.0)


@dataclass
class DataTrainingArguments:
    train_file:Optional[str] = field(default=None)
    valid_file:Optional[str] = field(default=None)
    test_file:Optional[str] = field(default=None)
    train_size:Optional[int] = field(default=None)
    max_train_samples:Optional[int] = field(default=None)
    max_valid_samples:Optional[int] = field(default=None)
    max_test_samples:Optional[int] = field(default=None)
    augment:Optional[str] = field(default=None)
    dataset_name:Optional[str] = field(default="causal-news")

    def __post_init__(self):
        if self.train_file is None and self.valid_file is None:
            raise ValueError("Need a training a file")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

 

    def model_init():
        init_args = {}
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_pretrained_checkpoint, num_labels=2)
        model.resize_token_embeddings(len(train_dataset.tokenizer))

        # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)
    train_dataset = ContrastiveClassificationDataset(path=data_args.train_file, dataset_type="train", size=None, 
                    tokenizer=model_args.tokenizer, max_length=model_args.max_length, dataset=data_args.dataset_name, aug=data_args.augment, frac=model_args.dataset_fraction)
    valid_dataset = ContrastiveClassificationDataset(path=data_args.valid_file, dataset_type="valid", size=None, 
                    tokenizer=model_args.tokenizer, max_length=model_args.max_length, dataset=data_args.dataset_name, aug=data_args.augment)
    if data_args.test_file:
        test_dataset = ContrastiveClassificationDataset(path=data_args.test_file, dataset_type="test", size=None, 
                        tokenizer=model_args.tokenizer, max_length=model_args.max_length, dataset=data_args.dataset_name, aug=data_args.augment)

    data_collator = DataCollatorClassification(tokenizer=train_dataset.tokenizer, max_length=model_args.max_length)
    callback = EarlyStoppingCallback(early_stopping_patience=10)


    output_dir = deepcopy(training_args.output_dir)
    for run in range(1):
        init_args = {}
        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f"{output_dir}"

        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        import pdb; pdb.set_trace()
        model = ContrastiveClassifierModel(len_tokenizer=len(train_dataset.tokenizer), checkpoint_path=model_args.model_pretrained_checkpoint, model='roberta-base', pool=True, comb_fct=None, frozen=True, pos_neg=False)
        import pdb; pdb.set_trace()

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset=train_dataset,
            eval_dataset = valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_bce
            # callbacks=[callback]
        )

        if training_args.do_train:
            # if model_args.do_param_opt:
            #     for n, v in best_run.hyperparameters.items():
            #         setattr(trainer.args, n, v)
            checkpoint = None 
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            train_dataset.tokenizer.save_pretrained(training_args.output_dir)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.log_metrics(f"train", metrics)
            trainer.save_metrics(f"train", metrics)
            trainer.save_state()
        
        results = {}
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(
                eval_dataset=valid_dataset,
                metric_key_prefix="eval"
            )
            max_eval_samples = len(valid_dataset)
            metrics["eval_samples"] = max_eval_samples

            trainer.log_metrics(f"eval", metrics)
            trainer.save_metrics(f"eval", metrics)


    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(
            test_dataset,
            metric_key_prefix="predict"
        )
        import pdb; pdb.set_trace()
        predictions = predictions[0]
        print(predictions)
        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_causal_news.json")
        with open(output_predict_file, "w") as writer:
            for index, item in enumerate(predictions):
                dict_data = {"index": index, "prediction": item.item()}
                writer.write(f"{json.dumps(dict_data)}\n")
    return results

if __name__ == '__main__':
    main()


            

