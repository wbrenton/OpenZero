# https://github.com/google/brax/blob/main/notebooks/Brax_v2_Training_Preview.ipynb ported to a script

import os
import functools
import jax

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

import flax
from brax.v2 import envs
from brax.training.agents import train as ppo

env_name = "ant"

env = envs.get_environment(env_name=env_name,
                           backend='generalized')

state = env.reset(rng=jax.random.PRNGKey(seed=0))

assert flax.struct.dataclasses.is_dataclass(env.sys)

train_fn = functools.partial(ppo.train, num_timesteps=25000000, num_evals=20,
                             reward_scaling=10, episode_length=1000,
                             normalize_observations=True, action_repeat=1,
                             unroll_length=5, num_minibatches=32,
                             num_updates_per_batch=4, discounting=0.97,
                             learning_rate=3e-4, entropy_cost=1e-2,
                             num_envs=2048, batch_size=1024)

max_y = 8000
min_y = 0

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  plt.xlim([0, train_fn.keywords['num_timesteps']])
  plt.ylim([min_y, max_y])
  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  
  # Create a "charts" folder if it doesn't exist
  os.makedirs('charts', exist_ok=True)
  
  # Specify the path to the output file
  chart_path = os.path.join('charts', 'chart.png')
  
  # Save the chart to the specified path
  plt.savefig(chart_path)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')