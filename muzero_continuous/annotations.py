import chex
import optax
import haiku as hk
from flax import struct

@struct.dataclass
class TrainState:
    params: hk.Params
    target_params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    train_step: chex.Array

@struct.dataclass
class TimeStep:
    observed_reward: chex.Array
    predicted_reward: chex.Array
    predicted_value: chex.Array
    search_policy: chex.Array
    predicted_policy: chex.Array
    discount: chex.Array

@struct.dataclass
class NetworkOutput:
    embedding: chex.Array
    next_embedding: chex.Array
    reward: chex.Array
    value: chex.Array
    policy: chex.Array

@struct.dataclass    
class NetworkApplys:
    muzero_network: callable
    representation: callable
    dynamics: callable
    prediction: callable