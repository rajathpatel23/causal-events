import numpy as np
np.random.seed(42)

import random
random.seed(42)

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

@dataclass
class DataCollatorContrastivePretrainSelfSupervised:
    tokenizer:PreTrainedTokenizerBase
    max_length:Optional[int] = 128
    pad_to_multiple_of:Optional[int] = None
    return_tensors: str="pt"

    def __call__(self, input):
        features_left = [x[0]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]
        
        batch = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']

        batch['labels'] = torch.LongTensor(labels)
        
        return batch





@dataclass
class DataCollatorContrastivePretrain:
    tokenizer:PreTrainedTokenizerBase
    max_length:Optional[int] = 128
    pad_to_multiple_of:Optional[int] = None
    return_tensors: str="pt"

    def __call__(self, input):
        features_left = [x[0]['features'] for x in input]
        features_right = [x[1]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)
        return batch


@dataclass
class DataCollatorContrastivePretrainCausalNews:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors:str = "pt"

    def __call__(self, input_both):
        rnd = random.choice([0, 1])
        input = [x[rnd] for x in input_both]
        feature_left = [x[0]['features'] for x in input]
        feature_right = [x[0]['features'] for x in input]
        labels = [x[0]['labels'] for x in input]
        batch_left = self.tokenizer(feature_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(feature_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)
        return batch

@dataclass
class DataCollatorContrastivePretrainCausalNewsSSV:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors:str = "pt"

    def __call__(self, input):
        feature_left = [x[0]['features'] for x in input]
        feature_right = [x[0]['features'] for x in input]

        labels = [x[0]['labels'] for x in input]

        batch_left = self.tokenizer(feature_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(feature_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)

        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)
        return batch


@dataclass
class DataCollatorContrastiveClassifcation:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors:str = "pt"

    def __call__(self, input):
        features_left = [x['features_left'] for x in input]
        features_right = [x['features_right'] for x in input]
        labels = [x['labels'] for x in input]
        
        batch_left = self.tokenizer(features_left, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        batch_right = self.tokenizer(features_right, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        
        batch = batch_left
        if 'token_type_ids' in batch.keys():
            del batch['token_type_ids']
        batch['input_ids_right'] = batch_right['input_ids']
        batch['attention_mask_right'] = batch_right['attention_mask']

        batch['labels'] = torch.LongTensor(labels)

        return batch

@dataclass
class DataCollatorClassification:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors:str = "pt"

    def __call__(self, input):
        features = [x['features'] for x in input]
        labels = []
        batch = self.tokenizer(features, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        if  "token_type_ids" in batch.keys():
            del batch['token_type_ids']
        
        labels = [x.get("labels", 0) for x in input]
        batch["labels"] = torch.LongTensor(labels)    
        return batch

@dataclass
class DataCollatorClassificationTestData:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    return_tensors:str = "pt"

    def __call__(self, input):
        features = [x['features'] for x in input]
        batch = self.tokenizer(features, padding=True, truncation=True, max_length=self.max_length, return_tensors=self.return_tensors)
        if  "token_type_ids" in batch.keys():
            del batch['token_type_ids']
        batch["labels"] = [0 for x in input]
        return batch

