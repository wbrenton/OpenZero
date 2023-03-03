import jax
import jax.numpy as jnp

from utils import make_action_representation_fn, make_action_discretization_fns
from mcts import make_mcts_fn
from network import make_muzero_network, make_apply_fns

observation_size = 87
action_size = 8
action_dim_support = 7

action_represenation_fn = make_action_representation_fn(action_size, 7)
muzero_network = make_muzero_network()

batch_of_obs = jnp.zeros((128, observation_size), dtype=jnp.float32)
batch_of_actions = jnp.zeros((128, action_size), dtype=jnp.int32)

actions = jnp.array([[0, 1, 2, 3, 4, 5, 6, 0] for _ in range(128)]) # 8 actions between 0 and 6
_, discrete_to_continuous_action_fn = make_action_discretization_fns(action_dim_support, -1., 1.)
continuous_action = discrete_to_continuous_action_fn(actions)
print(continuous_action.shape)
# batch_of_action_embed = action_represenation_fn(batch_of_actions)[:, 0, :]

# params, state = muzero_network.init(jax.random.PRNGKey(42), batch_of_obs, batch_of_action_embed)

# apply = make_apply_fns(muzero_network.apply, action_size, action_dim_support)

# network_output = apply.muzero_network(params, state, batch_of_obs, batch_of_actions)

# mcts_fn = make_mcts_fn(apply, action_size, action_dim_support, 10, 1000, 0.99)

# action, improved_policy = mcts_fn(params, state, batch_of_obs, 0, jax.random.PRNGKey(42))

# action representation
# action_representation_fn = make_action_representation_fn(8, 7)
# actions = jnp.array([[0, 1, 2, 3, 4, 5, 6, 0] for _ in range(128)]) # 8 actions between 0 and 6
# action_embedding = action_representation_fn(actions)
# print(action_embedding.shape)