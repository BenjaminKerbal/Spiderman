



import time
import gym
import numpy as np
from Agents.QAgents.StateAgents.StateAgent import StateAgent
from Environment.SpidermanEnv import SpidermanEnv
from ParallelHandler.ParallelEnvironments import ParallelEnvironments
from Utils.Enums import Agents
from Utils.Enums.ExplorationTypes import ExplorationTypes
from Utils.ExplorationHandler import ExplorationHandler
from Utils.TrainingProgressHandler import TrainingProgressHandler


class Trainer:

    RESTART_TEXT = "restart"
    RESTART_OBSERVATION = "obs"

    def __init__(self, agent_type, paralell_training, max_episodes_or_timesteps, train, render=False, print_rate=5, model=None, save_name=None, fail_safe=False):
        self.processes = 8
        self.envs_per_process = 4
        self.paralell_training = paralell_training
        if paralell_training:
            self.agent_action_batch = self.processes*self.envs_per_process
            self.env = ParallelEnvironments(SpidermanEnv, self.processes, self.envs_per_process)
            print_rate = 1000
            mean_print_values = print_rate
        else:
            self.agent_action_batch = 1
            self.env = SpidermanEnv(not train, render)
            mean_print_values = 10
        env_sizes = (self.env.observation_space, self.env.action_space)
        self.__create_agent(agent_type, env_sizes, train, save_name, self.agent_action_batch)
        self.train = train
        self.render = render
        if model is not None:
            self.agent.load_model(model)

        self.fail_safe = fail_safe
        self.last_episode_or_timestep = 0
        self.exploration = ExplorationHandler(ExplorationTypes.Gile, max_episodes_or_timesteps, train, 0.01, model is not None)
        self.training_progress_handler = TrainingProgressHandler(True, paralell_training, print_rate=print_rate, mean_episodes=mean_print_values)


    def __create_agent(self, agent_type, env_sizes, train, save_name, agent_action_batch_size):
        if agent_type == Agents.StateAgent:
            self.agent = StateAgent(env_sizes[0], env_sizes[1], not train, save_name, agent_action_batch_size=agent_action_batch_size)
        else:
            raise("not implemented")

    def restart_environment(self):
        self.agent.reset(True)
        if self.paralell_training:
            try:
                self.env.shutDownMultiprocessing()
            except:
                pass
            time.sleep(3)
            self.env = None
            time.sleep(1)
            self.env = ParallelEnvironments(SpidermanEnv, self.processes, self.envs_per_process, False)
        else:
            self.env = SpidermanEnv(not self.train, self.render)
        return self.env.reset()

    def reset(self):
        if not self.paralell_training:
            self.agent.reset()
        return self.env.reset().reshape(self.agent_action_batch, -1)

    def get_actions(self, obs, episode_or_timestep):
        self.last_episode_or_timestep = episode_or_timestep
        second_action_value = None
        epsilon = self.exploration.get_exploration(episode_or_timestep)
        actions = self.agent.get_action(obs, epsilon)
        if isinstance(actions, tuple):
            second_action_value = actions[1]
            actions = actions[0]
        return actions, second_action_value

    def step(self, actions):
        if isinstance(actions, tuple):
            actions = actions[0]
        if isinstance(actions, list) and not self.paralell_training:
            actions = actions[0]
        if self.fail_safe:
            try:
                observations, rewards, done, info = self.env.step(actions)
            except Exception as e:
                print("Error occured, restarting environments")
                obs = self.restart_environment()
                info = {
                    'obs' : obs,
                    'restart' : True
                }
                return -1, -1, -1, info
        else:
            observations, rewards, done, info = self.env.step(actions)
        return observations, rewards, done, info


    ''' Agent handlers '''
    def store_transition(self, *args):
        self.agent.store_transition(*args)

    def step_update(self):
        return self.agent.step_update()
    
    def end_of_episode_update(self):
        return self.agent.end_of_episode_update()
    
    def save_model(self, backup=False):
        if not (self.render or not self.train):
            self.agent.save_model(backup)

    ''' Training progress handlers '''
    def push_step_info(self, *args):
        self.training_progress_handler.push_step_info(*args)
    
    def push_end_of_episode_info(self, *args):
        self.training_progress_handler.push_end_of_episode_info(*args)

    def print(self):
        self.training_progress_handler.print()

    