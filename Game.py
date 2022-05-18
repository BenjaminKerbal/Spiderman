import time
import traceback

from Utils.Enums.Agents import Agents
from Utils.Trainer import Trainer



def train_agents(trainer, max_episodes_or_timesteps, train):
    if not trainer.paralell_training:
        time_step = 0
        for episode_or_timestep in range(max_episodes_or_timesteps):
            done = False
            obs = trainer.reset()
            while not done:
                actions = trainer.get_actions(obs, episode_or_timestep)
                new_obs, reward, done, info = trainer.step(actions)
                trainer.store_transition(obs, new_obs, actions, reward, done)
                loss = trainer.step_update()
                obs = new_obs
                trainer.push_step_info(episode_or_timestep, loss, reward, time_step, done, info)
                time_step += 1
                
            loss = trainer.end_of_episode_update()
            trainer.push_end_of_episode_info(episode_or_timestep, loss, reward, time_step, done, info)
            trainer.print()
    else:
        obs = trainer.reset()
        for episode_or_timestep in range(max_episodes_or_timesteps):
            actions = trainer.get_actions(obs, episode_or_timestep)
            new_obs, reward, env_done, info = trainer.step(actions)
            if trainer.RESTART_TEXT in info:
                obs = info[trainer.RESTART_OBSERVATION]
                continue
            trainer.store_transition(obs, new_obs, actions, reward, env_done)
            loss = trainer.step_update()
            obs = new_obs
            trainer.push_step_info(episode_or_timestep, loss, reward, episode_or_timestep, env_done, info)
            trainer.print()
        trainer.env.shutDownMultiprocessing()

    if train:
        trainer.save_model()


def start_game(agent_type, parallel_training, max_episodes_or_timesteps, train, render=False, catch_error_mode=True, model=None, save_name=None):
    render = True if not train else render
    parallel_training = False if render else parallel_training
    trainer = Trainer(agent_type, parallel_training, max_episodes_or_timesteps, train, render, model=model, save_name=save_name, fail_safe=catch_error_mode)
    start_time = time.time()
    if catch_error_mode:
        try:
            train_agents(trainer, max_episodes_or_timesteps, train)
        except:
            if train:
                trainer.agent.save_model(True)
            # trainer.env.shutDownMultiprocessing()
            print(traceback.format_exc())
    else:
        train_agents(trainer, max_episodes_or_timesteps, train)
    time_taken = time.time() - start_time
    print("Time taken (min):", round(time_taken / 60, 2))
    


if __name__ == '__main__':
    model = None
    model = 'BypassTuning'
    # model = 'test'
    start_game(Agents.StateAgent, 
        parallel_training=False, 
        max_episodes_or_timesteps= 100, #700000 ~= 17 h
        train=False, 
        render=True, 
        catch_error_mode=False,
        model=model, 
        save_name='test',
    )