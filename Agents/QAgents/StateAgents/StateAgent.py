
from functools import total_ordering
import numpy as np
from numpy import isin
import pandas as pd
from Agents.ParentAgent import ParentAgent

import random
import torch
import os
from os.path import join
import time
import torch.nn.functional as F
from numpy.random import choice

from Agents.QAgents.StateAgents.Policies.LinnearDueling import LinnearDueling
from Agents.QAgents.StateAgents.Policies.RecurrentDueling import RecurrentDueling
from Agents.QAgents.StateAgents.Policies.RecurrentDuelingBypass import RecurrentDuelingBypass
from Agents.QAgents.StateAgents.ReplayMemory import ReplayMemory, Transition



class StateAgent(ParentAgent):

    __metaclass__ = ParentAgent


    def __init__(self, observation_space, action_space, eval_mode, save_name=None, agent_action_batch_size=1):
        self.train_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.train_device = 'cpu'
        self.observation_space = observation_space
        self.a_space = action_space
        self.eval_mode = eval_mode
        self.agent_action_batch_size = agent_action_batch_size
        self.batch_size = 512 # 8 * agent_action_batch_size
        self.memory_size_before_training = 80000
        self.update_iterations = 2
        self.gamma = 0.96
        memorySize = int(1e6)
        self.replay_memory = ReplayMemory(memorySize)
        self.arg_max_chance = 1
        self.history_length = 30
        
        self.policy = RecurrentDuelingBypass(observation_space, self.a_space, self.history_length).to(self.train_device)
        self.policy.train()
        self.target_net = RecurrentDuelingBypass(observation_space, self.a_space, self.history_length).to(self.train_device)
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=2e-4)
        self.update_target_net_intervall = int(50000 / agent_action_batch_size)
        self.update_policy_interval = 1
        self.current_target_net_value = 0
        self.current_update_policy_value = 0
        self.state_train_shape = (self.batch_size, self.history_length, self.observation_space)
        self.agent_name = save_name
        self.reset(True)


    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def reset(self, initial=False):
        if self.agent_action_batch_size == 1 or initial:
            self.game_state = torch.zeros((self.agent_action_batch_size, self.history_length, self.observation_space)).to(self.train_device)
            self.save_state = torch.zeros((self.agent_action_batch_size, self.history_length, self.observation_space)).cpu()
            self.next_save_state = torch.zeros((self.agent_action_batch_size, self.history_length, self.observation_space)).cpu()

    def __update_target_network(self):
        self.target_net.load_state_dict(self.policy.state_dict())


    def __update_state_matrix(self, state_matrix, new_state):
        for i in range(state_matrix.shape[0]):
            state_matrix[i] = torch.roll(state_matrix[i], self.history_length - 1, dims=0)
            state_matrix[i][-1] = new_state[i,:]
        return state_matrix

    def __clean_up_states(self, ob, gpu_if_possible):
        ob = torch.from_numpy(ob).float().view(self.agent_action_batch_size, 1, -1)
        if gpu_if_possible:
            ob = ob.to(self.train_device)
        else:
            ob = ob.cpu()
        return ob

    def __get_any_action(self, actions, full_stack, epsilon):
        full_obs = self.game_state[full_stack, :]
        if full_obs.shape[0] != 0 and random.random() > epsilon:
            action_values = self.policy.forward(full_obs, True)
            if random.random() < self.arg_max_chance or self.eval_mode:
                actions[full_stack] = torch.argmax(action_values, dim=1).type(torch.int).cpu()
            else:
                available_actions = list(range(self.a_space))
                actions[full_stack] = torch.tensor([random.choices(available_actions, weights=action_values[i].tolist(), k=1)[0] for i in range(full_obs.shape[0])]).type(torch.int)
        actions[~full_stack] = 0
        return actions.tolist()

    '''  '''
    def __get_action_limited(self, actions, full_stack, epsilon, limit_actions):
        if limit_actions.sum() != 0:
            new_actions = (torch.rand(limit_actions.sum()) * 2).type(torch.int)
            actions[limit_actions] = new_actions
        if random.random() < epsilon:
            action_values = self.policy.forward(self.game_state, True)
            if random.random() < self.arg_max_chance or self.eval_mode:
                limited_action_values = action_values[limit_actions,:2]
                actions[limit_actions] = torch.argmax(limited_action_values, dim=1).type(torch.int).cpu()
                non_limited_action_values = action_values[~limit_actions]
                actions[~limit_actions] = torch.argmax(non_limited_action_values, dim=1).type(torch.int).cpu()
            else:
                limited_actions = list(range(1))
                non_limited_actions = list(range(self.a_space))

                limited_action_values = action_values[limit_actions,:2] # Divide by denumerator
                actions[limit_actions] = torch.tensor([random.choices(limited_actions, weights=limited_action_values[i].tolist(), k=1)[0] for i in range(limited_action_values.shape[0])]).type(torch.int)
                non_limited_action_values = action_values[~limit_actions]
                actions[~limit_actions] = torch.tensor([random.choices(non_limited_actions, weights=non_limited_action_values[i].tolist(), k=1)[0] for i in range(non_limited_action_values.shape[0])]).type(torch.int)
        actions[~full_stack] = 0
        return actions.tolist()


    def get_action(self, ob, epsilon, limit_actions=None):
        ob = self.__clean_up_states(ob, True)
        self.game_state = self.__update_state_matrix(self.game_state, ob)
        full_stack = torch.sum(self.game_state, axis=-1)[:,0] != 0
        with torch.no_grad():
            actions = (torch.rand(self.agent_action_batch_size) * self.a_space).type(torch.int)
            if limit_actions is not None:
                if isinstance(limit_actions, bool):
                    limit_actions = [limit_actions]
                limit_actions = torch.tensor(limit_actions).to(self.train_device)
                return self.__get_action_limited(actions, full_stack, epsilon, limit_actions)
            else:
                return self.__get_any_action(actions, full_stack, epsilon)

    def step_update(self):
        if self.eval_mode:
            return
        if len(self.replay_memory) < self.memory_size_before_training:
            return None
        
        self.current_target_net_value += 1
        if self.current_target_net_value > self.update_target_net_intervall:
            self.__update_target_network()
            self.current_target_net_value = 0
        
        self.current_update_policy_value += 1
        if self.current_update_policy_value < self.update_policy_interval:
            return None
        self.current_update_policy_value = 0

        total_loss = 0
        for _ in range(self.update_iterations):
            samples = self.replay_memory.sample(self.batch_size)
            total_loss += self.__do_batch_update(samples)
        total_loss = total_loss / self.update_iterations
        return total_loss

    def __do_batch_update(self, samples):
        samples = Transition(*zip(*samples))
        state = torch.stack(samples.state).float().view(self.state_train_shape).to(self.train_device).detach()
        next_state = torch.stack(samples.next_state).float().view(self.state_train_shape)

        dones = torch.stack(samples.dones)
        actions = torch.stack(samples.actions).to(self.train_device).type(torch.int64).view(-1, 1)
        rewards = torch.stack(samples.rewards).view(-1).to(self.train_device)

        non_final_mask = (1-dones.type(torch.float)).type(torch.bool)
        non_final_next_states_indexes = [i for i, nonfinal in enumerate(non_final_mask) if nonfinal > 0]
        non_final_next_states = next_state[non_final_next_states_indexes].squeeze(1).to(self.train_device)

        #Calulate action values with current policy and nextActionValues with target_net
        actionValues = self.policy.forward(state)
        currentActionValue = actionValues.gather(1, actions).squeeze(1)
        nextActionValues = torch.zeros(actions.shape[0]).to(self.train_device)
        nextActionValues[non_final_mask] = self.target_net.forward(non_final_next_states).max(1)[0].detach()
        targets = (rewards + self.gamma * nextActionValues).to(self.train_device)

        loss = F.smooth_l1_loss(currentActionValue, targets)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()
        return loss.item()

    def expects_one_action_value(self):
        return True

    def store_transition(self, state, next_state, actions, rewards, dones):
        if self.eval_mode:
            return
        if isinstance(actions, tuple):
            actions = actions[0]

        state = self.__clean_up_states(state, False)
        next_state = self.__clean_up_states(next_state, False)
        self.save_state = self.__update_state_matrix(self.save_state, state)
        self.next_save_state = self.__update_state_matrix(self.next_save_state, next_state)
        dones = torch.tensor(dones).cpu().byte()
        rewards = torch.tensor(rewards).cpu().type(torch.float32)
        actions = torch.tensor(actions).cpu()

        if self.agent_action_batch_size == 1:
            if torch.sum(self.save_state, axis=-1)[:,0] != 0:
                self.push_to_memory(self.save_state.view(-1), 
                                    self.next_save_state.view(-1), 
                                    actions, rewards, dones)

        else:
            full_stack = torch.sum(self.save_state, axis=-1)[:,0] != 0
            state_to_store = self.save_state[full_stack].cpu()
            next_state_to_store = self.next_save_state[full_stack].cpu()
            actions_to_store = actions[full_stack]
            rewards_to_store = rewards[full_stack]
            dones_to_store = dones[full_stack]
            for i in range(state_to_store.shape[0]):
                self.push_to_memory(state_to_store[i].view(-1), 
                                    next_state_to_store[i].view(-1), 
                                    actions_to_store[i], rewards_to_store[i], dones_to_store[i])

            if torch.sum(dones) != 0:
                # This should be on GPU
                self.game_state[dones==True] = torch.zeros((self.history_length, self.observation_space)).to(self.train_device)
                # These must be on CPU to not take up GPU memory
                self.save_state[dones==True] = torch.zeros((self.history_length, self.observation_space)).cpu()
                self.next_save_state[dones==True] = torch.zeros((self.history_length, self.observation_space)).cpu()

    def push_to_memory(self, *args):
        self.replay_memory.push(*args)










    

    
    

    