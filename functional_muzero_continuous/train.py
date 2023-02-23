import jax
import chex
import optax
import haiku as hk
import jax.numpy as jnp
from datetime import datetime
from brax.envs import wrappers as wrapper
from brax import envs as envs_v2

from loss import make_loss_fn
from mcts import make_mcts_fn
from network import make_muzero_network, make_apply_fns
from annotations import TrainState, TimeStep
from utils import make_broadcast_across_action_dims_fn, unpmap, make_action_discretization_fns

def train(num_timesteps = 10_000_000_000,
          batch_size = 32,
          num_envs = 1024,
          action_repeat = 1,
          td_steps=5,
          unroll_length = 10,
          episode_length = 1000,
          eval_length = 100,
          num_epochs = 100_000,
          observation_size = 27,
          action_size = 8,
          action_dim_support = 7,
          num_simulations = 50,
          discount=0.99,
          target_update_period=100,
          num_eval_episodes=10,):

    process_count = jax.process_count()
    device_count = jax.device_count()
    local_devices = jax.local_devices()

    # calculate number of training steps per epoch
    env_steps_per_training_step = (batch_size * unroll_length * action_repeat)
    num_train_steps_per_epoch = -(-num_timesteps // (num_epochs * env_steps_per_training_step))
    total_train_steps = num_train_steps_per_epoch * num_epochs

    # setup environment
    rng_seq = hk.PRNGSequence(42)
    env = envs_v2.get_environment("ant")
    env = wrapper.wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)

    eval_reset_fn = jax.jit(env.reset)
    reset_fn = jax.jit(jax.vmap(env.reset))

    ### TODO: create obs normalization with brax.running_statistics.normalize

    muzero_network_tx = make_muzero_network()
    apply = make_apply_fns(muzero_network_tx.apply, action_size, action_dim_support)
    optim = optax.adam(10e-4)
    
    broadcast_across_action_dim = make_broadcast_across_action_dims_fn(action_size)
    
    loss_fn = make_loss_fn(td_steps)
    loss_per_action_dim_fn = jax.vmap(loss_fn, in_axes=(1))

    mcts_fn = make_mcts_fn(apply, action_size, action_dim_support, num_simulations, total_train_steps, discount)
    
    _, discrete_to_continuous_action_fn = make_action_discretization_fns(action_dim_support, -1., 1.)

    def rollout_fn(params, target_params, state, env_state, train_step, rng):

        def step_fn(env_state, rng):
            discrete_action, improved_policy = mcts_fn(target_params, state, env_state.obs, train_step, rng)
            continuous_action = discrete_to_continuous_action_fn(discrete_action)
            preds, _ = apply.muzero_network(params, state, env_state.obs, discrete_action)
            chex.assert_equal_shape([improved_policy, preds.policy])
            chex.assert_equal_shape([preds.reward, preds.value, discrete_action])
            next_env_state = env.step(env_state, continuous_action)
            observed_reward = broadcast_across_action_dim(next_env_state.reward)
            discount = broadcast_across_action_dim(1. - next_env_state.done)
            return next_env_state, TimeStep(
                predicted_reward=preds.reward,
                predicted_value=preds.value,
                predicted_policy=preds.policy,
                observed_reward=observed_reward,
                search_policy=improved_policy,
                discount=discount)
            
        step_rngs = jax.random.split(rng, unroll_length)
        next_env_state, trajectories = jax.lax.scan(step_fn, env_state, step_rngs)
        reward = trajectories.observed_reward.mean(2).mean(1).sum(0)

        # calculate losses for each timestep per action dim
        reward_loss, policy_loss, value_loss, value_target = loss_per_action_dim_fn(trajectories)
        loss = reward_loss + policy_loss + value_loss
        loss = loss.mean() # mean across batch
        return loss, (next_env_state, {
            'scalar_loss': loss,
            'scalar_reward_loss': reward_loss.mean(),
            'scalar_policy_loss': policy_loss.mean(),
            'scalar_value_loss': value_loss.mean(),
            'scalar_reward': reward.mean(),
            'scalar_reward_pred': trajectories.predicted_reward.mean(2).mean(1).sum(0),
            'scalar_obs_reward': trajectories.observed_reward.mean(2).mean(1).sum(0),
            'scalar_value_pred': trajectories.predicted_value.mean(2).mean(1).sum(0),
            'scalar_value_target': value_target.mean(2).mean(1).sum(0),
            'dist_policy_pred': trajectories.predicted_policy[0,0],
            'dist_policy_target': trajectories.search_policy[0,0],
        })

    grad_fn = jax.grad(rollout_fn, has_aux=True)

    def update_fn(ts: TrainState, env_state, rng: jnp.ndarray):
        grads, aux = grad_fn(ts.params, ts.target_params, ts.state, env_state, ts.train_step, rng)
        next_env_state, metrics = aux
        grads = jax.lax.pmean(grads, axis_name='i')
        updates, new_opt_state = optim.update(grads, ts.opt_state)
        new_params = optax.apply_updates(ts.params, updates)
        new_train_step = ts.train_step + 1
        target_params = optax.periodic_update(new_params, ts.target_params, new_train_step, target_update_period)
        ts = TrainState(new_params, target_params, ts.state, new_opt_state, new_train_step)
        return (ts, next_env_state), metrics

    def training_epoch_fn(train_state: TrainState, env_state, rng: jnp.ndarray):
        
        def train_step(carry, rng):
            train_state, env_state = carry
            carry, metrics = update_fn(train_state, env_state, rng)
            return carry, metrics

        step_rngs = jax.random.split(rng, num_train_steps_per_epoch)
        (train_state, env_state), metrics = jax.lax.scan(train_step, (train_state, env_state), step_rngs)
        return train_state, env_state, metrics
    
    def make_eval_fn():
        
        def eval_fn(train_state, rng):
        
            def eval_rollout_fn(env_state, rng):
                action, _ = mcts_fn(params, state, env_state.obs, total_train_steps, rng)
                action = discrete_to_continuous_action_fn(action)
                next_env_state = env.step(env_state, action)
                return next_env_state, {
                    'scalar_reward': next_env_state.reward.mean(),
                    'action': action[0],
                }

            train_state = unpmap(train_state)
            params, state = train_state.params, train_state.state
            reset_rng = jax.random.split(rng, num_eval_episodes)
            env_states = eval_reset_fn(reset_rng)
            
            step_rngs = jax.random.split(rng, eval_length) #episode_length)
            _, metrics = jax.lax.scan(eval_rollout_fn, env_states, step_rngs)
            return metrics
        
        return jax.jit(eval_fn)
    
    training_epoch_fn = jax.pmap(training_epoch_fn, axis_name='i')
    
    obs = jnp.zeros((1, observation_size))
    action_embed = jnp.zeros((1, action_size * action_dim_support))
    params, state = muzero_network_tx.init(next(rng_seq), obs, action_embed)
    opt_state = optim.init(params)
    train_state = TrainState(params=params,
                             target_params=params,
                             state=state,
                             opt_state=opt_state,
                             train_step=0)

    train_state = jax.device_put_replicated(train_state, devices=local_devices)

    key_envs = jax.random.split(next(rng_seq), batch_size // process_count)
    key_envs = jnp.reshape(key_envs, (device_count, -1) + key_envs.shape[1:])
    env_states = reset_fn(key_envs)

    # initial evaluation
    eval_fn = make_eval_fn()
    metrics = eval_fn(train_state, next(rng_seq))
    print(f'epoch: {0}, initial_reward: {metrics["scalar_reward"].sum()} action: {metrics["action"][0]}')

    rngs = jax.random.split(next(rng_seq), device_count)
    t = datetime.now()
    for epoch in range(1, num_epochs + 1):
        train_state, env_states, train_metrics = training_epoch_fn(train_state, env_states, rngs)
        train_metrics = jax.tree_map(lambda x: x.mean(0), train_metrics)
        train_metrics = jax.tree_map(lambda x: x.block_until_ready(), train_metrics)

        if epoch % 10 == 0:
            d_t = datetime.now() - t
            metrics = eval_fn(train_state, next(rng_seq))
            train_steps = train_state.train_step[0]
            print(f'epoch: {epoch}, train steps: {train_steps} loss: {train_metrics["scalar_loss"].mean()}, reward: {train_metrics["scalar_reward"].mean()} time: {d_t} test_reward: {metrics["scalar_reward"].sum()}')
            t = datetime.now()

        rngs = jax.random.split(next(rng_seq), device_count)

# look at how brax did it, there may be something to preconfiguring tthe params to be an arg to scan
#with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):

env = envs_v2.get_environment("ant")
train(observation_size=env.observation_size,
      action_size=env.action_size)