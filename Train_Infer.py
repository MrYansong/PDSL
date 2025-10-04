# coding: utf-8
import torch.nn as nn


class TrainingModel(nn.Module):
    def __init__(self, cvae_model: nn.Module, propagate: nn.Module,  mlp_model: nn.Module):
        super(TrainingModel, self).__init__()

        self.cvae_model = cvae_model
        self.mlp_model = mlp_model
        self.propagate = propagate

    def forward(self, input_pair, condition):  # (batch_size, nodes_num)
        seed_hat, mean, log_var = self.cvae_model(input_pair, condition)
        seed_hat_tmp = seed_hat.unsqueeze(-1)
        predictions = self.propagate(seed_hat_tmp)
        predictions = self.mlp_model(predictions)
        predictions = predictions.squeeze(-1)
        return seed_hat, mean, log_var, predictions

class InferenceModel(nn.Module):
    def __init__(self, mlp_model: nn.Module, propagate: nn.Module):
        super(InferenceModel, self).__init__()
        self.mlp_model = mlp_model
        self.propagate = propagate
        self.relu = nn.ReLU(inplace=True)

    def forward(self, seed_vec):
        seed_vec = seed_vec.unsqueeze(-1)
        predictions = self.propagate(seed_vec)
        predictions = self.mlp_model(predictions)
        return predictions
