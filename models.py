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
                 aggr='mean'
                 ):
        super(Siamese, self).__init__()
        self.language_model = AutoModel.from_pretrained(model_name)  # Load language model from HuggingFace
        self.aggr = Aggregation(aggr)  # Type of word vector aggregation to generate sentence embeddings
        self.linear = nn.Linear(self.language_model.config.hidden_size * 3, 3)  # Linear layer post concatenation

    def forward(self, premise, hypothesis):

        encoded_premise = self.language_model(premise)[0]
        aggregated_premise = self.aggr(encoded_premise)

        encoded_hypothesis = self.language_model(hypothesis)[0]
        aggregated_hypothesis = self.aggr(encoded_hypothesis)

        difference = torch.abs(aggregated_premise - aggregated_hypothesis)

        concatenation = torch.cat([aggregated_premise, aggregated_hypothesis, difference], dim=1)

        return self.linear(concatenation)  # return logits
