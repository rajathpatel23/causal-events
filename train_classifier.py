import logging

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

from copy import deepcopy

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, AutoModelForSequenceClassification
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers import EarlyStoppingCallback
from transformers.utils.hp_naming import TrialShortNamer
from src.data.datasets import ContrastiveClassificationDataset
from src.data.data_collators import DataCollatorClassification

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

    def get_posneg():
        if data_args.dataset_name == "causal-news":
            return 4

    def model_init():
        init_args = {}
        pos_neg = get_posneg()
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
                    tokenizer=model_args.tokenizer, max_length=256, dataset=data_args.dataset_name, aug=data_args.augment)
    valid_dataset = ContrastiveClassificationDataset(path=data_args.valid_file, dataset_type="valid", size=None, 
                    tokenizer=model_args.tokenizer, max_length=256, dataset=data_args.dataset_name, aug=data_args.augment)
    if data_args.test_file:
        test_dataset = ContrastiveClassificationDataset(path=data_args.test_file, dataset_type="test", size=None, 
                        tokenizer=model_args.tokenizer, max_length=256, dataset=data_args.dataset_name, aug=data_args.augment)

    data_collator = DataCollatorClassification(tokenizer=train_dataset.tokenizer)
    callback = EarlyStoppingCallback(early_stopping_patience=10)

    if training_args.do_train and model_args.do_param_opt:
        from ray import tune
        def my_hp_space(trial):
            return {
                "learning_rate": tune.loguniform(5e-5, 5e-3),
                "warmup_ratio": tune.choice([0.05, 0.075, 0.10]),
                "max_grad_norm": tune.choice([0.0, 1.0]),
                "weight_decay": tune.loguniform(0.001, 0.1),
                "seed": tune.randint(1, 50)
            }
        
        def my_objective(metrics):
            return metrics['eval_f1']

        trainer = Trainer(model=model_init, args=training_args, train_dataset=train_dataset, 
        eval_dataset=valid_dataset, test_dataset=test_dataset if data_args.test_file is not None else None, 
        data_collator=data_collator, 
        compute_metrics=compute_metrics_bce, 
        callbacks=[callback])

        def hp_name(trial):
            namer = TrialShortNamer()
            namer.set_defaults('hp', {'learning_rate': 1e-4, 'warmup_ratio': 0.0, 'max_grad_norm': 1.0, 'weight_decay': 0.01, 'seed':1})
            return namer.shortname(trial)

        initial_configs = [
            {
                "learning_rate": 1e-3,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            },
            {
                "learning_rate": 1e-4,
                "warmup_ratio": 0.05,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "seed": 42
            }
            ]

        from ray.tune.suggest.hebo import HEBOSearch
        hebo = HEBOSearch(metric="eval_f1", mode="max", points_to_evaluate=initial_configs, random_state_seed=42)
        best_run = trainer.hyperparameter_search(n_trials=24, direction="maximize", hp_space=my_hp_space, compute_objective=my_objective, backend='ray', resources_per_trail={'cpu':2,'gpu':1}, 
        local_dir=f'{training_args.output_dir}ray_results/', hp_name=hp_name, search_alg=hebo)
        with open(f"{training_args.output_dir}best_params.json", "w") as f:
            json.dump(best_run, f)

    output_dir = deepcopy(training_args.output_dir)
    for run in range(3):
        init_args = {}
        training_args.save_total_limit = 1
        training_args.seed = run
        training_args.output_dir = f"{output_dir}"

        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        pos_neg = get_posneg()
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_pretrained_checkpoint, num_labels=2)
        model.resize_token_embeddings(len(train_dataset.tokenizer))

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset=train_dataset,
            eval_dataset = valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_soft_max,
            callbacks=[callback]
        )

        if training_args.do_train:
            if model_args.do_param_opt:
                for n, v in best_run.hyperparameters.items():
                    setattr(trainer.args, n, v)

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
                metric_key_prefix="eval"
            )
            max_eval_samples = len(valid_dataset)
            metrics["eval_samples"] = max_eval_samples

            trainer.log_metrics(f"eval", metrics)
            trainer.save_metrics(f"eval", metrics)

        if training_args.do_predict:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                test_dataset,
                metric_key_prefix="predict"
            )

            metrics = predict_results.metrics
            max_predict_samples = len(test_dataset)
            metrics["predict_samples"] = max_predict_samples

            trainer.log_metrics(f"predict", metrics)
            trainer.save_metrics(f"predict", metrics)
    return results

if __name__ == '__main__':
    main()




    
            

