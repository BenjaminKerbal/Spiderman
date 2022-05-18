
import torch
import torch.nn as nn
import torch.nn.functional as F




class LinnearDueling(torch.nn.Module):

    def __init__(self, observation_space, action_space, history_length=1, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(observation_space * history_length, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear_value = nn.Linear(hidden_size // 2, 1)
        self.linear_advantage = nn.Linear(hidden_size // 2, action_space)

    def forward(self, x, return_advantage=False):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        A = self.linear_advantage(x)
        if return_advantage:
            return F.softmax(A, dim=1)
        V = self.linear_value(x)
        Q = V + A - A.mean(1, keepdims=True)
        return Q