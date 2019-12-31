import torch.nn as nn
import torch
from transformers import AutoModel


class Aggregation(nn.Module):
    """
    Helper class to perform aggregation (default mean) over the word embeddings
    """
    def __init__(self, aggr='mean'):
        super(Aggregation, self).__init__()
        self.aggr = aggr
        assert self.aggr == 'mean' or 'max' or 'CLS', "aggr must be one of: {mean, max, CLS}"

    def forward(self, x):
        if self.aggr == 'mean':
            return x.mean(dim=1)
        if self.aggr == 'max':
            return x.max(dim=1)[0]
        if self.aggr == 'CLS':
            return x[:, 0, :]


class Siamese(nn.Module):
    """
    Class which carries out the Siamese network training
    """
    def __init__(self,
                 model_name='distilbert-base-uncased',
                 num_labels=4,
                 aggr='mean',
                 device='cuda'):
        super(Siamese, self).__init__()
        self.language_model = AutoModel.from_pretrained(model_name)  # Load language model from HuggingFace
        self.num_labels = num_labels
        self.aggr = Aggregation(aggr)  # Type of word vector aggregation to generate sentence embeddings
        self.device = device
        self.linear = nn.Linear(self.language_model.config.hidden_size * 3,
                                self.num_labels)  # Linear layer post concatenation
        self.to(device)  # Send entire model to GPU if available

    def forward(self, batch):
        premise = batch.premise
        hypothesis = batch.hypothesis

        premise = self.language_model(premise)[0]
        premise = self.aggr(premise)

        hypothesis = self.language_model(hypothesis)[0]
        hypothesis = self.aggr(hypothesis)

        concatenation = torch.cat([premise, hypothesis, torch.abs(premise - hypothesis)], dim=1)

        return self.linear(concatenation)  # return logits
