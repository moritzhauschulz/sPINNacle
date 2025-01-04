import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Sequence, Optional
import chex
from optax._src import base

class MultiAdamState(NamedTuple):
    """State for the MultiAdam optimizer."""
    step: chex.Array
    exp_avg: Sequence[base.Updates]  # One for each group
    exp_avg_sq: Sequence[base.Updates]  # One for each group
    max_exp_avg_sq: Sequence[base.Updates]  # One for each group
    agg_exp_avg: base.Updates
    agg_exp_avg_sq: base.Updates

def multiadam(
    learning_rate: float = 1e-3,
    betas: tuple = (0.99, 0.99),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    loss_group_idx: Optional[Sequence[int]] = None,
    group_weights: Optional[Sequence[float]] = None,
    agg_momentum: bool = False,
    agg_betas: Optional[tuple] = None,
    amsgrad: bool = False,
    maximize: bool = False,
) -> optax.GradientTransformation:
    """MultiAdam optimizer matching PyTorch implementation exactly."""
    
    n_groups = 2 if loss_group_idx is None else len(loss_group_idx) + 1
    group_weights = jnp.ones(n_groups) / n_groups if group_weights is None else jnp.array(group_weights)
    
    if agg_momentum and agg_betas is None:
        raise ValueError('agg_betas should be provided when agg_momentum is True')
    
    if agg_momentum:
        agg_beta1, agg_beta2 = agg_betas
    else:
        agg_beta1, agg_beta2 = 0.0, 0.0

    def init_fn(params):
        exp_avg = [jax.tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
        exp_avg_sq = [jax.tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
        max_exp_avg_sq = [jax.tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
        
        return MultiAdamState(
            step=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            max_exp_avg_sq=max_exp_avg_sq,
            agg_exp_avg=jax.tree_map(jnp.zeros_like, params),
            agg_exp_avg_sq=jax.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates, state, params=None):
        step = state.step + 1
        beta1, beta2 = betas
        
        def update_group(g, exp_avg_g, exp_avg_sq_g, max_exp_avg_sq_g, weight):
            # Fix: Apply negation inside tree_map when maximize is True
            grad = jax.tree_map(lambda x: (-x if not maximize else x), g)
            
            # Apply weight decay before momentum if specified
            if weight_decay != 0 and params is not None:
                grad = jax.tree_map(
                    lambda g, p: g + weight_decay * p,
                    grad, params)

            # Update biased first and second moment estimates
            exp_avg_g = jax.tree_map(
                lambda m, g: beta1 * m + (1 - beta1) * g,
                exp_avg_g, grad)
            
            exp_avg_sq_g = jax.tree_map(
                lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
                exp_avg_sq_g, grad)
            
            if amsgrad:
                max_exp_avg_sq_g = jax.tree_map(
                    lambda m, v: jnp.maximum(m, v),
                    max_exp_avg_sq_g, exp_avg_sq_g)
            else:
                max_exp_avg_sq_g = exp_avg_sq_g
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # Compute step size
            step_size = learning_rate / bias_correction1
            
            # Compute update
            group_updates = jax.tree_map(
                lambda m, v: (m / (jnp.sqrt(v / bias_correction2) + eps)) * weight,
                exp_avg_g,
                max_exp_avg_sq_g)
            
            return group_updates, (exp_avg_g, exp_avg_sq_g, max_exp_avg_sq_g)
        
        # Process each group
        all_updates = []
        new_states = []
        for i in range(n_groups):
            group_update, (new_exp_avg, new_exp_avg_sq, new_max_exp_avg_sq) = update_group(
                updates, state.exp_avg[i], state.exp_avg_sq[i],
                state.max_exp_avg_sq[i], group_weights[i])
            all_updates.append(group_update)
            new_states.append((new_exp_avg, new_exp_avg_sq, new_max_exp_avg_sq))
        
        # Sum updates across groups
        final_updates = jax.tree_map(lambda *x: sum(x), *all_updates)
        
        # Apply aggregate momentum if specified
        if agg_momentum:
            bias_correction1 = 1 - agg_beta1 ** step
            bias_correction2 = 1 - agg_beta2 ** step
            
            # Update aggregate moments
            agg_exp_avg = jax.tree_map(
                lambda m, u: agg_beta1 * m + (1 - agg_beta1) * u,
                state.agg_exp_avg, final_updates)
            
            agg_exp_avg_sq = jax.tree_map(
                lambda v, u: agg_beta2 * v + (1 - agg_beta2) * jnp.square(u),
                state.agg_exp_avg_sq, final_updates)
            
            # Apply aggregate momentum correction
            final_updates = jax.tree_map(
                lambda m, v: (m / bias_correction1) / (jnp.sqrt(v / bias_correction2) + eps),
                agg_exp_avg, agg_exp_avg_sq)
        else:
            agg_exp_avg = state.agg_exp_avg
            agg_exp_avg_sq = state.agg_exp_avg_sq
        
        # Apply learning rate
        final_updates = jax.tree_map(lambda x: learning_rate * x, final_updates)
        
        return final_updates, MultiAdamState(
            step=step,
            exp_avg=[s[0] for s in new_states],
            exp_avg_sq=[s[1] for s in new_states],
            max_exp_avg_sq=[s[2] for s in new_states],
            agg_exp_avg=agg_exp_avg,
            agg_exp_avg_sq=agg_exp_avg_sq
        )
    
    return optax.GradientTransformation(init_fn, update_fn)