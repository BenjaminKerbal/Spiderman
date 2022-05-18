import multiprocessing as mp
import random
import gym
import numpy as np

from ParallelHandler.SerialEnvironment import SerialEnvironment



class ParallelEnvironments(object):

    RESTART_STRING = "restart"
    BASE_SEED = 123

    def __init__(self, environment_class, processes, envs_per_process, set_mp_context=True):
        self.parent_pipes = []
        self.processes = []
        self.env_name = environment_class
        self.num_processes = processes
        self.envs_per_process = envs_per_process
        if set_mp_context:
            mp.set_start_method('spawn')
        for i in range(self.num_processes):
            parent_pipe, child_pipe = mp.Pipe()
            process = mp.Process(target=ParallelEnvironments.worker_proc, args=(environment_class, child_pipe, i, envs_per_process))
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

        for p in self.processes:
            p.start()

        temp_env = environment_class()
        self.action_space = temp_env.action_space
        self.observation_space = temp_env.observation_space
        self.limit_actions = np.array([False] * processes * envs_per_process) 

    @staticmethod
    def worker_proc(environment, pipe, worker_id, num_envs):
        env = SerialEnvironment(environment, num_envs)
        np.random.seed(ParallelEnvironments.BASE_SEED + worker_id)
        random.seed(ParallelEnvironments.BASE_SEED + worker_id)
        while True:
            cmd, actions = pipe.recv()
            if cmd == "step":
                try:
                    response = env.step(actions)
                    pipe.send((response))
                except:
                    pipe.send((-1, -1, -1, {ParallelEnvironments.RESTART_STRING : True}))
            elif cmd == "reset":
                pipe.send((env.reset()))
            elif cmd == "quit":
                # print("Worker quitting.")
                break
            else:
                raise ValueError("Unrecognized command:", cmd)

    def step(self, actions):
        totalReward = []
        idx, inc = 0, self.envs_per_process
        actions = np.array(actions)
        # Send step commands
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("step", actions[idx:(idx+inc)]))
            idx += inc

        # read resulting states
        obs = []
        rewards = []
        dones = []
        totalReward = []
        total_score = []
        # limit_actions = []
        for i in range(self.num_processes):
            ob, reward, done, info = self.parent_pipes[i].recv()
            if self.RESTART_STRING in info:
                raise Exception("Multiprocess step failed")
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            totalReward.extend(info['totalReward'])
            total_score.extend(info['total_score'])
            # limit_actions.extend(info['limit_actions'])

        obs = np.concatenate(obs, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        infos = {
            'totalReward' : totalReward,
            'total_score' : total_score,
        }
        dones = np.concatenate(dones, axis=0)
        # self.limit_actions = np.array(limit_actions)
        return obs, rewards, dones, infos

    def reset(self):
        # Send reset commands
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("reset", 0))

        # Read states
        obs = []
        for i in range(self.num_processes):
            obs.append(self.parent_pipes[i].recv())
        obs = np.concatenate(obs, axis=0)
        return obs

    def shutDownMultiprocessing(self):
        for i in range(self.num_processes):
            try:
                self.parent_pipes[i].send(("quit", 0))
            except:
                pass
