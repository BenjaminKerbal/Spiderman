from collections import OrderedDict
import copy
from statistics import mean
import time

import numpy as np


class TrainingProgressHandler:

    def __init__(self, step_save: bool,  paralell_training, print_rate=5, mean_episodes=None) -> None:
        self.paralell_training = paralell_training
        self.step_save = step_save
        self.print_rate = print_rate
        self.mean_episodes = print_rate if mean_episodes is None else mean_episodes
        self.memory_step = OrderedDict()
        self.memory_episode = OrderedDict()
        self.memory_timestep = []
        self.last_episode = -1
        self.last_print_timestep = 0
        self.round_decimals = 4
        self.parallel_data_dict = {}
        self.parallel_envs_done = 0

    def __clean_up_dict(self):
        cur_dict = self.memory_step[self.last_episode]
        loss_list = [loss for loss in cur_dict['loss'] if loss is not None]
        if len(loss_list) != 0:
            cur_dict['loss'] = round(sum(loss_list) / len(loss_list), self.round_decimals + 2)
        else:
            cur_dict['loss'] = None
        if 'env_steps' in cur_dict:
            cur_dict['env_steps'] = max(cur_dict['env_steps'])
        if 'placed_bombs' in cur_dict:
            cur_dict['placed_bombs'] = max(cur_dict['placed_bombs'])
        if 'barrels_destroyed' in cur_dict:
            cur_dict['barrels_destroyed'] = max(cur_dict['barrels_destroyed'])
        if 'train_agent_won' in cur_dict:
            del cur_dict['train_agent_won']

    def __save_non_parallel_info(self, episode, loss, reward, info):
        # if self.step_save:
        self.last_episode = episode
        if not episode in self.memory_step:
            info_dict = {
                'episode' : episode,
                'reward' : [reward],
                'loss' : [loss]
            }
            info = {key : [value] for key, value in info.items()}
            info_dict.update(info)
            self.memory_step[episode] = info_dict
        else:
            cur_dict = self.memory_step[episode]
            cur_dict['loss'].append(loss)
            cur_dict['reward'].append(reward)
            # for key, value in info.items():
            #     cur_dict[key].append(value)

    def __save_parallel_info(self, timestep, loss, env_done, info):
        loss = np.nan if loss is None else loss
        self.parallel_envs_done += np.sum(env_done)
        total_rewards = np.array(info['totalReward']).mean() if len(info['totalReward']) != 0 else np.nan
        total_score = np.array(info['total_score']).mean() if len(info['total_score']) != 0 else np.nan
        info_dict = {
            'steps' : timestep + 1,
            'games' : self.parallel_envs_done,
            'reward' : total_rewards,
            'score' : total_score,
            'loss' : loss
        }
        
        self.memory_timestep.append(info_dict)
        return

        if len(self.parallel_data_dict) == 0:
            self.parallel_keys = info['infos'][0].keys()
            self.temp_dict = {key : [] for key in self.parallel_keys}
            self.parallel_data_dict = {i: copy.deepcopy(self.temp_dict) for i in range(env_done.shape[0])}

        for i, dict_list in enumerate(info['infos']):
            for key, value in dict_list.items():
                self.parallel_data_dict[i][key].append(value)

        if np.sum(env_done) == 0:
            avg_info = {key: np.nan for key in self.parallel_keys}
            avg_info['reward'] = np.nan
        else:
            finished_data = np.array([np.array(list(self.parallel_data_dict[i].values())).max(axis=1) for i in range(len(self.parallel_data_dict)) if env_done[i]])
            mean_values = np.mean(finished_data, axis=0)
            avg_info = {key: mean_values[i] for i, key in enumerate(self.parallel_keys)}
            avg_info['reward'] = mean([value[0] for value in info['totalReward']])
            self.parallel_data_dict = {i: copy.deepcopy(self.temp_dict) if env_done[i] else dict_values for i, dict_values in self.parallel_data_dict.items()}
        info_dict.update(avg_info)
        

    def push_step_info(self, episode, loss, reward, current_timestep, env_done, info):
        if self.step_save:
            if self.paralell_training:
                self.__save_parallel_info(episode, loss, env_done, info)
            else:
                self.__save_non_parallel_info(episode, loss, reward, info)


    def push_end_of_episode_info(self, episode, loss, reward, current_timestep, env_done, info):
        if not self.step_save:
            if self.paralell_training:
                self.__save_parallel_info(episode, loss, reward, env_done, info)
            else:
                self.__save_non_parallel_info(episode, loss, reward, info)
        return

    def __print_non_parallel(self):
        self.__clean_up_dict()
        if self.last_episode % self.print_rate == 0 and self.last_episode != 0:
            results_dict = {key: [] for key in list(self.memory_step[0].keys())}
            dict_keys_list = list(self.memory_step.keys())[-self.mean_episodes:]
            for episode_key in dict_keys_list:
                temp_dict = self.memory_step[episode_key]
                for key in results_dict.keys():
                    value = temp_dict[key]
                    if value is not None:
                        if isinstance(value, list):
                            value = sum(value)
                        results_dict[key].append(value)

            for key, value_list in results_dict.items():
                if len(value_list) == 0:
                    value = None
                elif len(value_list) == 1 or key == 'episode' or key == 'current_timestep':
                    value = value_list[-1]
                else:
                    value = round(sum(value_list) / len(value_list), self.round_decimals)
                print(key + ": " + str(value) + " ", end='')
            print()

    def __print_parallel(self):
        self.last_print_timestep += 1
        if self.last_print_timestep < self.print_rate:
            return
        self.last_print_timestep = 0

        data = self.memory_timestep[-self.mean_episodes:]
        keys_list = list(data[0].keys())
        temp_dict = {key : [] for key in keys_list}
        for dict_list in data:
            for key, value in dict_list.items():
                temp_dict[key].append(value)
        
        for key, value_list in temp_dict.copy().items():
            if key in ['steps', 'games']:
                temp_dict[key] = np.array(value_list).max()
            else:
                value_list = np.array(value_list)
                if np.isnan(value_list).all():
                    temp_dict[key] = np.nan
                else:
                    temp_dict[key] = np.nanmean(value_list)
        
        np_array = np.round(np.array([values for values in temp_dict.values()]), self.round_decimals)
        for i in range(np_array.shape[0]):
            print(keys_list[i] + ": " + str(np_array[i]) + "  | ", end='')
        print()

    def print(self):
        if self.paralell_training:
            self.__print_parallel()
        else:
            self.__print_non_parallel()
