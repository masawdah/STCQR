import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# Create sequences per station
def create_sequences(df, variables, n_steps=15):
    X, y, stations, dates = [], [], [], []
    for st, df_st in df.groupby('station'):
        data = df_st[variables].reset_index(drop=True)
        date_col = df_st['datetime']
        for i in range(len(data) - n_steps):
            X.append(data.iloc[i:i+n_steps, :-1].values)  # input features
            y.append(data.iloc[i+n_steps, -1])            # next target
            stations.append(st)
            dates.append(date_col.iloc[i+n_steps]) 
    return np.array(X), np.array(y), np.array(stations), np.array(dates)


class PinballLoss(nn.Module):
    __name__ = "Pinball"

    def __init__(
        self,
        tau: float,
    ):
        """
        Pinball Loss for regression tasks.

        Parameters:
        tau: Quantile level (0 < tau < 1).
        target_weight: Dictionary of targets with associated weights.
                       In the form {target: weight}.
        """
        super(PinballLoss, self).__init__()
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1")
        self.tau = tau

    def forward(self, y_true, y_pred):
        """
        Calculate the Pinball Loss between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Returns:
        torch.Tensor: The Pinball loss.
        """
        error = y_true - y_pred
        loss = torch.maximum(self.tau * error, (self.tau - 1) * error)

        return loss.mean()

# Define LSTM Model
class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, quantiles):
        super().__init__()
        self.hidden_size = hidden_size
        self.quantiles = quantiles

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(quantiles))  # one output per quantile

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # use last timestep
        return out  # shape: (batch, n_quantiles)



