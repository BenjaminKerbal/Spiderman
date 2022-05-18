
import random

from collections import namedtuple
import numpy as np
import pandas as pd
import torch
from os.path import join

Transition = namedtuple('Transition', ('state', 'next_state', 'actions', 'rewards', 'dones'))

class ReplayMemory(object):
    def __init__(self, capacity, prioritize_recent=False):
        self.capacity = capacity
        self.prioritize_recent = prioritize_recent
        self.memory = []
        self.position = 0

    def __prioritize_recent_sampling(self, batch_size):
        if len(self.memory) < 8 * batch_size or self.position < len(self.memory) < 8:
            return random.sample(self.memory, batch_size)
        
        recent_batch_size = batch_size // 4
        old_batch_size = batch_size - recent_batch_size
        if self.position < recent_batch_size*8:
            recent_remaining = recent_batch_size*8 - self.position
            recent_samples = self.memory[:self.position] + self.memory[-recent_remaining:]
            older_samples = self.memory[self.position:-recent_remaining]
        else:
            recent_start = self.position - recent_batch_size*8
            recent_samples = self.memory[recent_start:self.position]
            older_samples = self.memory[:recent_start] + self.memory[self.position:]
        sample_recent = random.sample(recent_samples, recent_batch_size)
        sample_old = random.sample(older_samples, old_batch_size)
        return sample_old + sample_recent

    def __even_sampling(self, batch_size):
        if len(self.memory) < batch_size:
            return
        return random.sample(self.memory, batch_size)


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.prioritize_recent:
            return self.__prioritize_recent_sampling(batch_size)
        else:
            return self.__even_sampling(batch_size)

    def __len__(self):
        return len(self.memory)


    def get_header_names(self, state_space_grid, state_space_pos):
        headerNames = ['state_grid_' + str(i+1) for i in range(state_space_grid)]
        headerNames = headerNames + ['state_pos_' + str(i+1) for i in range(state_space_pos)]
        headerNames = headerNames + ['next_state_grid_' + str(i+1) for i in range(state_space_grid)]
        headerNames = headerNames + ['next_state_pos_' + str(i+1) for i in range(state_space_pos)]
        return headerNames + ['action', 'reward', 'done']

    def save_to_dataframe(self, location, save_name='DefaultName', previous_df=None):
        all_transitions = Transition(*zip(*self.memory))
        header_names = self.get_header_names(all_transitions.state_grid[0].shape[0], all_transitions.state_pos[0].shape[0])
        df = pd.DataFrame(
            np.concatenate((
            torch.stack(all_transitions.state_grid).numpy(),
            torch.stack(all_transitions.state_pos).numpy(),
            torch.stack(all_transitions.next_state_grid).numpy(), 
            torch.stack(all_transitions.next_state_pos).numpy(),
            torch.stack(all_transitions.actions).reshape(-1, 1).numpy(), 
            torch.stack(all_transitions.rewards).reshape(-1, 1).numpy(),
            torch.stack(all_transitions.dones).reshape(-1, 1).numpy()
        ), axis=1), 
        columns=header_names).round(5)
        save_name = save_name if save_name.endswith('.csv') else save_name + ".csv"
        if previous_df is not None:
            df = pd.concat([df, previous_df])
        df.to_csv(join(location, save_name), index=False)



    @staticmethod
    def load_from_dataframe(file_data, file_data_as_dataframe=False, desiredSize=500000):
        if file_data_as_dataframe:
            df = file_data
        else:
            df = pd.read_csv(file_data)
        state_space_grid = int(len([column for column in df.columns.values if 'state_grid_' in column]) / 2)
        state_space_pos = int(len([column for column in df.columns.values if 'state_pos_' in column]) / 2)
        start_i = 0
        end_i = state_space_grid
        state_grid =  torch.from_numpy(df.iloc[:, :state_space_grid].values).type(torch.float32)

        start_i = end_i
        end_i = start_i + state_space_pos
        state_pos =  torch.from_numpy(df.iloc[:, start_i:end_i].values).type(torch.float32)

        start_i = end_i
        end_i = start_i + state_space_grid
        next_state_grid = torch.from_numpy(df.iloc[:, start_i:end_i].values).type(torch.float32)

        start_i = end_i
        end_i = start_i + state_space_pos
        next_state_pos = torch.from_numpy(df.iloc[:, start_i:end_i].values).type(torch.float32)

        action = torch.from_numpy(df['action'].values).type(torch.float32)
        reward = torch.from_numpy(df['reward'].values).type(torch.float32)
        done =  torch.from_numpy(df['done'].values).type(torch.float32)
        memory_size = max(done.shape[0], desiredSize)
        replay_memory = ReplayMemory(memory_size)
        for i in range(done.shape[0]):
            replay_memory.push(state_grid[i], state_pos[i], next_state_grid[i], next_state_pos[i], action[i], reward[i], done[i])
        return replay_memory
    