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
    def __init__(self, path, tokenizer="roberta-base", max_length=128, 
                dataset='causal-news',
                aug=False) -> None:
        super().__init__()

        self.max_length=max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=[])
        self.dataset = dataset
        self.aug = aug
        data = pd.read_csv(path)
        cluster_id_set = [x for x in range(len(data))]
        data1 = data.copy()
        data1['cluster_id'] = cluster_id_set
        data2 = data.copy()
        data2['cluster_id'] = cluster_id_set
        data2 = data.copy()
        label_enc = LabelEncoder()
        label_enc.fit(cluster_id_set)
        data1['features'] = data1['text']
        data2['features'] = data2['text']
        data1['labels'] = label_enc.transform(data1['cluster_id'])
        data2['labels'] = label_enc.transform(data2['cluster_id'])
        self.label_encoder = label_enc

        data1 = data1.reset_index(drop=True)
        data1 = data1.fillna("")
        data1 = self._prepare_data(data1)
        self.data1 = data1
        self.data2 = data2


    def __getitem__(self, idx):
        example1 = self.data1.loc[idx].copy()
        selection1 = self.data1[self.data1['labels'] == example1['labels']]
        pos1 = selection1.sample(1).iloc[0].copy()

        example2 = self.data2.loc[idx].copy()
        selection2 = self.data2[self.data2['labels'] == example2['labels']]
        pos2 = selection2.sample(1).iloc[0].copy()

        return ((example1, pos1), (example2, pos2))

    def __len__(self):
        return len(self.data1)

    




        