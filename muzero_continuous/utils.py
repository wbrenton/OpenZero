import jax
import jax.numpy as jnp
import chex
import optax 
import haiku as hk
from flax import struct
import rlax
from rlax._src import nonlinear_bellman
from brax.envs import env
from datetime import datetime

unsqueeze = lambda x: jnp.expand_dims(x, axis=0)
squeeze = lambda x: x[0]
unpmap = lambda x: jax.tree_map(squeeze, x)

def softmax_temperature_fn(current_step, total_steps):
    """Used for annealing exploration in search throughout training.

    Args:
        current_step (int)
        total_steps (int)

    Returns:
        float: softmax temperature
    """
    percent_complete = current_step / total_steps
    return jax.lax.cond(
        jnp.less_equal(
            percent_complete,
            0.5),
        lambda: 1.,
        lambda: jax.lax.select(
            jnp.less_equal(
                percent_complete,
                0.75),
            0.5,
            0.25)
        )
def make_action_representation_inside_mcts(action_dim, action_dim_support):

    def action_representation_inside_mcts(action, action_idx):
    
        def _per_action_dim_action_representation(action, action_idx):
            action_embedding = jnp.zeros((action_dim, action_dim_support))
            action_embedding = action_embedding.at[action_idx, action].set(1)
            return action_embedding.flatten()

        return jax.vmap(_per_action_dim_action_representation, in_axes=(0, None))(action, action_idx)
    
    return action_representation_inside_mcts

def make_action_representation_fn(action_dim, action_dim_support):

    def action_representation_fn(actions):
        """Converts categorical actions to a one-hot action dimension dependent representation.

        Args:
            actions (jnp.ndarray): a batch of actions of shape (batch_size, action_dim) 
            where each value along action_dim is an integer in [0, action_dim_support)

        Returns:
            jnp.ndarray: a batch of action representations of shape 
                (batch_size, action_dim, (action_dim * action_dim_support))
        """

        def _action_representation_fn(actions):
            action_idxs = jnp.arange(action_dim)
            
            # indexing with arrays in not supported, this is a workaround
            def _per_action_dim_action_representation(action_idx, action):
                action_embedding = jnp.zeros((action_dim, action_dim_support))
                action_embedding = action_embedding.at[action_idx, action].set(1)
                return action_embedding.flatten()

            return jax.vmap(_per_action_dim_action_representation)(action_idxs, actions)
        
        return jax.vmap(_action_representation_fn)(actions)
    
    return action_representation_fn

def make_broadcast_across_action_dims_fn(action_dim):
    
    def broadcast_across_action_dims(array):
        """Broadcasts an array across action dimensions."""
        array = jnp.expand_dims(array, axis=0)
        return jnp.broadcast_to(array, (action_dim, *array.shape[1:])).swapaxes(1, 0)
    
    return broadcast_across_action_dims



# categorical representation of continuous action space
def make_action_discretization_fns(num_bins, min_val, max_val):
    tx = rlax.muzero_pair(num_bins=num_bins,
                        min_value=min_val,
                        max_value=max_val,
                        tx=nonlinear_bellman.IDENTITY_PAIR)
    
    
    def continuous_to_discrete_action_fn(continuous_action):
        return tx.apply(continuous_action)

    def discrete_to_continuous_action_fn(discrete_action):
        one_hot_action = jax.nn.one_hot(discrete_action, num_bins)
        return tx.apply_inv(one_hot_action)
    
    return continuous_to_discrete_action_fn, discrete_to_continuous_action_fn