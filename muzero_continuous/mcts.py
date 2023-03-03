import jax
import mctx
import jax.numpy as jnp

from utils import make_action_representation_inside_mcts, make_broadcast_across_action_dims_fn, softmax_temperature_fn

def make_mcts_fn(apply, action_size, action_dim_support, num_simulations, training_steps, discount):
    
    # TODO: change the dynamics and prediction to be not operating over action dims
    # TODO: expose the per action dim representation fn
    broadcast_across_action_dims_fn = make_broadcast_across_action_dims_fn(action_size)
    action_representation_fn = make_action_representation_inside_mcts(action_size, action_dim_support)
    action_dims = jnp.arange(action_size)
    
    def recurrent_fn(carry, rng_key, action, embedding):
            params_and_state, action_dim = carry
            action_embed = action_representation_fn(action, action_dim)
            (next_embedding, reward), _ = apply.dynamics(*params_and_state, embedding, action_embed)
            (policy, value), _ = apply.prediction(*params_and_state, next_embedding)
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.full_like(reward, discount), # is the optional
                prior_logits=policy,
                value=value)
            return (recurrent_fn_output, next_embedding)
        
    def per_action_dim_fn(params, state, embedding, train_step, rng, action_dim):
        (policy, value), _ = apply.prediction(params, state, embedding)
        root = mctx.RootFnOutput(
            prior_logits=policy,
            value=value,
            embedding=embedding)
        search_policy_output = mctx.muzero_policy(
            params=((params, state), action_dim),
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=num_simulations, 
            temperature=softmax_temperature_fn(train_step, training_steps))
        return search_policy_output

    per_action_dim_fn = jax.vmap(per_action_dim_fn, in_axes=(None, None, None, None, None, 0))

    def mcts_fn(params, state, obs, train_step, rng):
        """MCTS for each action dimension.
        Observations are mapped to embeddings with the representation network.
        These embeddings are then broadcasted to each action dimension.
        MCTS search is performed in each action dimension, with the root being the same embeddings across.
        'action_dims' is used to create a action dimension dependent action representation
        enabling the network to learn action dimension dependent mappings.
        The result of MCTS is a policy of size (action_dim, batch_size, N) 
        and actions of size (action_dim, batch_size, 1).
        """ 

        embedding, _ = apply.representation(params, state, obs)
        search_policy_output = per_action_dim_fn(params, state, embedding, train_step, rng, action_dims)
        improved_policy = search_policy_output.action_weights.squeeze().swapaxes(1,0)
        action = search_policy_output.action.squeeze().swapaxes(1,0)
        return action, improved_policy

    return mcts_fn
