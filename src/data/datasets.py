import logging
import numpy as np
from augmentation import delete_random_tokens
np.random.seed(42)

import random
random.seed(42)

import pandas as pd

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from sklearn.preprocessing import LabelEncoder



class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer="roberta-base", max_length=128, 
                intermediate_set=None, clean=False, dataset="causal_news", only_interim=False, 
                aug=False) -> None:
        super().__init__()

        self.max_length=max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=[])
        self.dataset = dataset
        self.aug = aug