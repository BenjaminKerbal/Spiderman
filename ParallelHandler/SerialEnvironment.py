import random
import time
import numpy as np




class SerialEnvironment(object):
    def __init__(self, environment_class, n_envs):
        self.envs = []
        self.n_envs = n_envs
        for i in range(n_envs):
            env = environment_class()
            self.envs.append(env)
        self.rewards = np.zeros(n_envs)

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = {}
        totalReward = []
        total_score = []
        # limit_actions = [False] * self.n_envs
        for i in range(self.n_envs):
            try:
                ob, reward, done, info = self.envs[i].step(actions[i])
            except:
                print("Error while stepping env. Resetting.")
                reward, done = 0, True
                info = { 'score' : 0 }
            self.rewards[i] += reward
            if done:
                total_score.append(info['score'])
                try:
                    ob = self.envs[i].reset()
                except:
                    print("Error while resetting env. Trying again")
                    ob = self.envs[i].reset()
                totalReward.append(self.rewards[i])
                self.rewards[i] = 0
            # limit_actions[i] = self.envs[i].limit_actions
            obs.append(ob.reshape(1,-1))
            rewards.append(reward)
            dones.append(done)
        
        infos = {
            'totalReward' : totalReward,
            'total_score' : total_score,
            # 'limit_actions' : limit_actions
        }
        obs = np.concatenate(obs, axis=0)
        rewards = np.stack(rewards)
        return obs, rewards, dones, infos

    def reset(self):
        obs = []
        for e in self.envs:
            obs.append(e.reset().reshape(1,-1))
        obs = np.concatenate(obs, axis=0)
        return obs