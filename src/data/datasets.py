
import numpy as np
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
                aug=False, dev=True) -> None:
        super().__init__()

        self.max_length=max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=['[CLS]', '[SEP'])
        self.dataset = dataset
        self.aug = aug
        data = pd.read_csv(path)
        if dev:
            dev_data = pd.read_csv("/home/jovyan/work/causal-events/data/subtask1/dev_subtask1.csv")
            data = pd.concat([data, dev_data])
        cluster_id_set = data['label'].tolist()
        data1 = data.copy()
        data1['cluster_id'] = cluster_id_set
        data2 = data.copy()
        data2 = data2.sample(frac = 1)
        data2['cluster_id'] = cluster_id_set
        label_enc = LabelEncoder()
        label_enc.fit(cluster_id_set)
        import pdb; pdb.set_trace()
        data1['features'] = data1['text'].str.lower()
        data2['features'] = data2['text'].str.lower()
        data1['labels'] = label_enc.transform(data1['cluster_id'])
        data2['labels'] = label_enc.transform(data2['cluster_id'])
        self.label_encoder = label_enc
        data1 = data1.reset_index(drop=True)
        data1 = data1.fillna("")
        data2 = data2.reset_index(drop=True)
        data2 = data2.fillna("")
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


class ContrastiveClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer="roberta-base", max_length=128, dataset='causal-news', aug=False, frac=1.0) -> None:

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=[])
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug
        self.frac = frac

        # if dataset == "causal-news":
        data = pd.read_csv(path)
        data = data.reset_index(drop=True)
        import pdb; pdb.set_trace()
        if self.frac != 1.0:
            data = data.sample(frac=self.frac)
        data['text']  = data['text'].str.lower()
        data = data.rename(columns={"text": "features", "label": "labels"})
        self.data = data

    def __getitem__(self, idx):
        example = self.data.iloc[idx].copy()
        return example

    def __len__(self):
        return len(self.data)
        

        
class ContrastiveClassificationTestData(torch.utils.data.Dataset):
    def __init__(self, path, dataset_type, size=None, tokenizer="roberta-base", max_length=128, dataset='causal-news', aug=False) -> None:

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=[])
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.aug = aug

        # if dataset == "causal-news":
        data = pd.read_csv(path)
        data = data.reset_index(drop=True)
        data = data.rename(columns={"text": "features"})
        self.data = data

    def __getitem__(self, idx):
        example = self.data.iloc[idx].copy()
        return example

    def __len__(self):
        return len(self.data)
        