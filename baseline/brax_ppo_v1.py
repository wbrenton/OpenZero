from brax import envs
from brax.training.agents.ppo import train as ppo
from brax import jumpy as jp
import functools
from datetime import datetime
import matplotlib.pyplot as plt

env_name = "ant"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']
env = envs.get_environment(env_name=env_name)
state = env.reset(rng=jp.random_prngkey(seed=0))

train_fn = functools.partial(ppo.train,
                             num_timesteps=25000000,
                             num_evals=20,
                             reward_scaling=10,
                             episode_length=1000,
                             normalize_observations=True,
                             action_repeat=1,
                             unroll_length=5,
                             num_minibatches=32,
                             num_updates_per_batch=4,
                             discounting=0.97,
                             learning_rate=3e-4,
                             entropy_cost=1e-2,
                             num_envs=2048,
                             batch_size=1024)

max_y = 8000
min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  print(f'{num_steps} steps: {metrics["eval/episode_reward"]:.2f} reward')

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)