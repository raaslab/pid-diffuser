import os
import collections
import numpy as np
import gymnasium as gym
import pdb

import gymnasium_robotics
# import minari ß
from torch.utils.data import DataLoader
import h5py
from .dataset import Dataset

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name, reset_target=True):
    if type(name) != str:
        ## name is already an environment
        return name
    with suppress_output():
        wrapped_env = gym.make('PointMaze_UMaze-v3', reset_target=reset_target)
        # wrapped_env = gym.make(name, reset_target=reset_target)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(name):
    return Dataset("diffuser/datasets/preloaded_data/pointmaze-umaze-sparse.hdf5")
    # return Dataset(os.path.join("preloaded_data", name))

# def load_dataset_and_environment(name): 
#     dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2')
#     env  = dataset.recover_environment(eval_env=True)

#     return dataset, env

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset.rewards.shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    # use_timeouts = 'timeouts' in dataset
    use_timeouts = True
 
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset.terminals[i])
        if use_timeouts:
            final_timestep = dataset.timeouts[i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset.mapping:
            if 'metadata' in k: continue
            data_[k].append(dataset.get(k)[i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
