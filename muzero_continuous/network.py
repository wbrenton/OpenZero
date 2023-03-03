import jax
import chex
import haiku as hk
import jax.numpy as jnp

from utils import make_action_representation_fn, make_broadcast_across_action_dims_fn
from annotations import NetworkOutput, NetworkApplys

nonlinearity = jax.nn.relu

def _torso(num_hiddens=512, num_layers=10):
    layers = []
    for _ in range(num_layers):
        layers += [hk.Linear(num_hiddens), nonlinearity]
    return hk.Sequential(layers)

class RepresentationNetwork(hk.Module):
    def __init__(self, num_layers, torso_hiddens, name='h'):
        super().__init__(name=name)
        self.torso = _torso(torso_hiddens, num_layers)

    def __call__(self, observation, is_training=True):
        h = hk.Flatten()(observation)
        latent_rep = self.torso(h)
        return latent_rep

class DynamicsNetwork(hk.Module):
    def __init__(self, num_layers, torso_hiddens, name='g'):
        super().__init__(name=name)
        self.torso = _torso(torso_hiddens, num_layers)
        self.reward_head = hk.Linear(1)

    def __call__(self, latent_rep, action_rep, is_training=True):
        latent_rep_concat_action_rep = jnp.append(latent_rep, action_rep, axis=-1)
        next_latent_rep = self.torso(latent_rep_concat_action_rep)
        reward = self.reward_head(next_latent_rep)
        return next_latent_rep, reward.squeeze(-1)

class PredictionNetwork(hk.Module):
    def __init__(self, torso_hiddens, action_dim_support, prediction_net_layers, name='f'):
        super().__init__(name=name)
        self.torso = _torso(torso_hiddens, prediction_net_layers)
        self.value_head = hk.Linear(1)
        self.policy_head = hk.Linear(action_dim_support)

    def __call__(self, latent_rep, is_training=True):
        h = self.torso(latent_rep)
        policy = self.policy_head(h)
        value = self.value_head(h)
        return policy, value.squeeze(-1)


def make_muzero_network(num_torso_layers=1,
                        torso_hidden_size=32,
                        action_dim_support=7, 
                        prediction_net_layers=1):
    """Creates a MuZero network along with subnetworks."""

    def fn():
        representation = RepresentationNetwork(num_torso_layers, torso_hidden_size)
        dynamics = DynamicsNetwork(num_torso_layers, torso_hidden_size)
        prediction = PredictionNetwork(torso_hidden_size, action_dim_support, prediction_net_layers)
        
        def h(observation):
            return representation(observation)
            
        def g(embedding, action_embed):
            return dynamics(embedding, action_embed)
            
        def f(embedding):
            return prediction(embedding)

        def init(observation, action_embed):
            """ This function is used to initialize the network. """
            chex.assert_rank([observation, action_embed], [2, 2]) 
            
            embedding = h(observation)
            next_embedding, reward = g(embedding, action_embed)
            policy, value = f(next_embedding)
            
            return NetworkOutput(embedding=embedding,
                                next_embedding=next_embedding,
                                reward=reward,
                                value=value,
                                policy=policy)

        return init, (init, h, g, f)

    return hk.without_apply_rng(hk.multi_transform_with_state(fn))

def make_apply_fns(applys, action_size, action_dim_support):
    action_representation_fn = make_action_representation_fn(action_size, action_dim_support)
    broadcast_across_action_dims = make_broadcast_across_action_dims_fn(action_size)
    
    # TODO: use different indexing method this looks dirty
    representation_apply = applys[1]
    dynamics_apply = applys[2]
    prediction_apply = applys[3]
    per_action_dynamics_apply = jax.vmap(dynamics_apply, in_axes=(None, None, 0, 0))
    per_action_prediction_apply = jax.vmap(prediction_apply, in_axes=(None, None, 0))

    def muzero_network_apply(params, state, observation, action):
        embedding, state = representation_apply(params, state, observation)
        embedding = broadcast_across_action_dims(embedding)
        action_rep = action_representation_fn(action)

        (next_embedding, reward), state = per_action_dynamics_apply(params, state, embedding, action_rep)
        (policy, value), state = per_action_prediction_apply(params, state, next_embedding)
        return NetworkOutput(embedding=embedding,
                             next_embedding=next_embedding,
                             reward=reward,
                             policy=policy,
                             value=value), state

    return NetworkApplys(muzero_network_apply,
                         representation_apply,
                         dynamics_apply,
                         prediction_apply) 