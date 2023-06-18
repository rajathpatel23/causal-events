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

    train_data = ContrastivePretrainDataset(path=train_path, tokenizer='roberta-base', max_length=128)
    valid_data = ContrastivePretrainDataset(path=valid_path, tokenizer='roberta-base', max_length=128)
    model = ContrastivePretrainModel(len_tokenizer=128, model='roberta-base', pool='True', proj='mlp', temperature=0.07)






