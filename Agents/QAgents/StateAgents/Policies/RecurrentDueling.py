
import torch
import torch.nn as nn
import torch.nn.functional as F




class RecurrentDueling(torch.nn.Module):
    
    def __init__(self, observation_space, action_space, history_length=1, hidden_size=128):
        super().__init__()


        # Save as LargerDueling
        self.gru = nn.GRU(observation_space, hidden_size * 2, 3, batch_first=True)
        self.linear_middle = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_advantage = nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, action_space)
        )
        self.linear_value = nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, 1)
        )
        

    def forward(self, x, return_advantage=False):
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = F.relu(self.linear_middle(x))
        A = self.linear_advantage(x)
        if return_advantage:
            return F.softmax(A, dim=1)
        V = self.linear_value(x)
        Q = V + A - A.mean(1, keepdims=True)
        return Q