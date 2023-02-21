import jax
from jax import vmap
import jax.numpy as jnp
from optax import l2_loss, softmax_cross_entropy
import haiku as hk
from optax import scale_gradient

def get_value_targets(rewards, values, discounts, n_steps, gamma=0.99):
    values = jax.lax.stop_gradient(values)
    exps = jnp.arange(n_steps+1)
    gamma = jnp.array(gamma)

    def get_value_target_k(k):
        discounts_k = jax.lax.dynamic_slice_in_dim(discounts, k, n_steps, axis=0)
        boundry_mask = get_episode_boundry_mask(discounts_k)
        gammas_k = jnp.power(gamma, exps) 
        rewards_k = jax.lax.dynamic_slice_in_dim(rewards, k, n_steps, axis=0)
        value_k_plus_n = jax.lax.dynamic_index_in_dim(values, k+n_steps-1, axis=0)
        bootstrap = jnp.concatenate([rewards_k, value_k_plus_n])
        discounted = gammas_k * bootstrap
        masked_output_vector = jax.lax.select(boundry_mask, discounted, jnp.zeros_like(discounted))
        return masked_output_vector.sum()

    ks = jnp.arange(n_steps)
    targets = jax.vmap(get_value_target_k)(ks)
    return targets

def make_loss_fn(td_steps):
    from_k_to_K = lambda x: jax.lax.slice_in_dim(x, 0, td_steps, axis=0)
    
    batched_value_target_fn = vmap(get_value_targets, in_axes=(1,1,1,None))

    def loss_fn(d):
        scaled_rewards = d.observed_reward * 0.01
        # reward
        reward_pred = from_k_to_K(d.predicted_reward)
        reward_target = from_k_to_K(scaled_rewards)
        reward_loss = l2_loss(reward_pred, jax.lax.stop_gradient(reward_target))

        # policy
        policy_pred = from_k_to_K(d.predicted_policy)
        policy_target = from_k_to_K(d.search_policy)
        # use of softmax_cross_entropy is possible bug
        policy_loss = softmax_cross_entropy(policy_pred, jax.lax.stop_gradient(policy_target))

        # value
        value_pred = from_k_to_K(d.predicted_value)
        value_target = batched_value_target_fn(
            scaled_rewards,
            d.predicted_value,
            d.discount,
            td_steps).swapaxes(1,0)
        value_loss = l2_loss(value_pred, jax.lax.stop_gradient(value_target))

        return reward_loss.mean(), policy_loss.mean(), value_loss.mean(), value_target

    return loss_fn

def get_episode_boundry_mask(discounts):
    
    def scan_fn(is_zero, discounts):
        is_zero = jax.lax.select(jnp.logical_or(is_zero, jnp.equal(discounts, 0)), True, False)
        mask = jax.lax.select(is_zero, 0, 1) 
        return is_zero, jnp.array(mask)
    
    is_zero = jnp.equal(discounts[0], 0)
    _, mask = jax.lax.scan(scan_fn, is_zero, discounts)
    return jnp.append(mask, mask[-1])

# discounts = jnp.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
# rewards = jnp.arange(10).astype(jnp.float32)
# values = jnp.arange(10).astype(jnp.float32)
# n_steps = 5

# targets = get_value_targets(rewards, values, discounts, n_steps)
