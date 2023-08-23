import os
import transformers
import torch
from torch import nn
from transformers import AutoModel
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

class SupConLoss(nn.Module):
    def __init__(self,temperature=0.07, contrast_model="all", base_temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.contrast_model = contrast_model
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device("cuda") if features.is_cuda else torch.device("cpu"))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_model == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_model == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss



class BaseEncoder(nn.Module):
    def __init__(self, len_tokenizer, model_name="roberta-base") -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer.resize_token_embeddings(len_tokenizer)
        self.model_name = model_name

    def forward(self, input_ids, attention_masks):
        output = self.transformer(input_ids, attention_masks)
        return output



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand((token_embeddings.size())).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class ContrastivePretrainModel(nn.Module):
    def __init__(self, len_tokenizer, model="roberta-base", pool=True, proj="mlp", temperature=0.07, logger=None) -> None:
        super().__init__()
        self.temperature = temperature
        self.pool = pool
        self.proj =  proj
        self.criterion = SupConLoss(self.temperature)
        self.encoder = BaseEncoder(len_tokenizer, model)
        self.logger = logger

    def forward(self, input_ids, attention_mask, labels, input_ids_right, attention_mask_right):
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask_right)

        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)
        output = F.normalize(output, dim=-1)
        loss = self.criterion(output, labels)
        return loss, output

class ContrastivePretrainHead(nn.Module):
    def __init__(self, hidden_size, proj='mlp'):
        super().__init__()
        if proj == 'linear':
            self.proj = nn.Linear(hidden_size, hidden_size)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, hidden_states):
        x = self.proj(hidden_states)
        return x


class ClassificationHead(nn.Module):

    def __init__(self, config, comb_fct):
        super().__init__()

        if comb_fct in ['concat-abs-diff', 'concat-mult']:
            self.hidden_size = 3 * config.hidden_size
        elif comb_fct in ['concat', 'abs-diff-mult']:
            self.hidden_size = 2 * config.hidden_size
        elif comb_fct in ['abs-diff', 'mult']:
            self.hidden_size = config.hidden_size
        elif comb_fct in ['concat-abs-diff-mult']:
            self.hidden_size = 4 * config.hidden_size
        else:
            self.hidden_size = config.hidden_size

        classifier_dropout = config.hidden_dropout_prob
        
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x



class ContrastiveClassifierModel(nn.Module):

    def __init__(self, len_tokenizer, checkpoint_path, model='huawei-noah/TinyBERT_General_4L_312D', pool=True, comb_fct='concat-abs-diff-mult', frozen=True, pos_neg=False):
        super().__init__()
        self.pool = pool
        self.frozen = frozen
        self.checkpoint_path = checkpoint_path
        self.comb_fct = comb_fct
        self.pos_neg = pos_neg

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config
        if self.pos_neg:
            self.criterion = BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_neg]))
        else:
            self.criterion = BCEWithLogitsLoss()
        self.classification_head = ClassificationHead(self.config, self.comb_fct)
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.load_state_dict(checkpoint, strict=False)
        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, labels):
        if self.pool:
            output = self.encoder(input_ids, attention_mask)
            output = mean_pooling(output, attention_mask)
        else:
            output = self.encoder(input_ids, attention_mask)['pooler_output']
        proj_output = self.classification_head(output)
        if labels is not None:
            loss = self.criterion(proj_output.view(-1), labels.float())
        else:
            loss = 0
        proj_output = torch.sigmoid(proj_output)
        return (loss, proj_output)


class ConstrastiveSelfSupervised(nn.Module):
    def __init__(self, len_tokenizer:int, model:str, pool:bool = True, proj:str ='mlp', temperature:float=0.07) -> None:
        super().__init__()
        self.pool = pool
        self.encoder = BaseEncoder(len_tokenizer=len_tokenizer, model=model)
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask, input_ids_right, attention_mask_right):
        if self.pool:
            output_left = self.encoder(input_ids, attention_mask)
            output_left = mean_pooling(output_left, attention_mask=attention_mask)

            output_right = self.encoder(input_ids_right, attention_mask_right)
            output_right = mean_pooling(output_right, attention_mask=attention_mask_right)
        else:
            output_left = self.encoder(input_ids, attention_mask)['pooler_output']
            output_right = self.encoder(input_ids_right, attention_mask_right)["pooler_output"]
        output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)))
        output = F.normalize(output, dim=-1)
        loss = self.criterion(output)
        return loss, output


class ContrastiveModel(nn.Module):
    def __init__(
        self,
        len_tokenizer,
        model="huawei-noah/TinyBERT_General_4L_312D",
        pool=True,
        proj="mlp",
        temperature=0.07,
    ):
        """Model used for application"""
        super().__init__()

        self.pool = pool
        self.proj = proj
        self.temperature = temperature
        self.criterion = SupConLoss(self.temperature)

        self.encoder = BaseEncoder(len_tokenizer, model)
        self.config = self.encoder.transformer.config

    def forward(self, input_ids, attention_mask):
        if self.pool:
            output = self.encoder(input_ids, attention_mask)
            output = mean_pooling(output, attention_mask)

            # output_right = self.encoder(input_ids_right, attention_mask_right)
            # output_right = mean_pooling(output_right, attention_mask_right)
        else:
            output = self.encoder(input_ids, attention_mask)["pooler_output"]
            # output_right = self.encoder(input_ids_right, attention_mask_right)['pooler_output']

        # output = torch.cat((output_left.unsqueeze(1), output_right.unsqueeze(1)), 1)

        output = F.normalize(output, dim=-1)

        # Do not calculate loss - only for application
        loss = 0

        return (loss, output)