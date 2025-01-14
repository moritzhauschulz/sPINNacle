import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_map

def init_multiadam_state(params, n_groups):
    exp_avg = [tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
    exp_avg_sq = [tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
    max_exp_avg_sq = [tree_map(jnp.zeros_like, params) for _ in range(n_groups)]
    agg_exp_avg = tree_map(jnp.zeros_like, params)
    agg_exp_avg_sq = tree_map(jnp.zeros_like, params)
    return {
        'step': jnp.zeros([], jnp.int32),
        'exp_avg': exp_avg,
        'exp_avg_sq': exp_avg_sq,
        'max_exp_avg_sq': max_exp_avg_sq,
        'agg_exp_avg': agg_exp_avg,
        'agg_exp_avg_sq': agg_exp_avg_sq
    }

def multiadam_update(grads, state, params, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize, group_weights, agg_momentum, agg_beta1, agg_beta2):
    step = state['step'] + 1
    exp_avg = state['exp_avg']
    exp_avg_sq = state['exp_avg_sq']
    max_exp_avg_sq = state['max_exp_avg_sq']
    agg_exp_avg = state['agg_exp_avg']
    agg_exp_avg_sq = state['agg_exp_avg_sq']

    def update_group(g, exp_avg_g, exp_avg_sq_g, max_exp_avg_sq_g, weight):
        grad = tree_map(lambda x: x if maximize else -x, g)
        if weight_decay != 0:
            grad = tree_map(lambda g, p: g + weight_decay * p, grad, params)
        exp_avg_g = tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, exp_avg_g, grad)
        exp_avg_sq_g = tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g), exp_avg_sq_g, grad)
        if amsgrad:
            max_exp_avg_sq_g = tree_map(lambda m, v: jnp.maximum(m, v), max_exp_avg_sq_g, exp_avg_sq_g)
        else:
            max_exp_avg_sq_g = exp_avg_sq_g
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = lr / bias_correction1
        group_updates = tree_map(lambda m, v: (m / (jnp.sqrt(v / bias_correction2) + eps)) * weight, exp_avg_g, max_exp_avg_sq_g)
        return group_updates, (exp_avg_g, exp_avg_sq_g, max_exp_avg_sq_g)

    all_updates = []
    new_states = []
    for i in range(len(grads)):
        group_update, (new_exp_avg, new_exp_avg_sq, new_max_exp_avg_sq) = update_group(grads[i], exp_avg[i], exp_avg_sq[i], max_exp_avg_sq[i], group_weights[i])
        all_updates.append(group_update)
        new_states.append((new_exp_avg, new_exp_avg_sq, new_max_exp_avg_sq))

    final_updates = tree_map(lambda *x: sum(x), *all_updates)

    if agg_momentum:
        bias_correction1 = 1 - agg_beta1 ** step
        bias_correction2 = 1 - agg_beta2 ** step
        agg_exp_avg = tree_map(lambda m, u: agg_beta1 * m + (1 - agg_beta1) * u, agg_exp_avg, final_updates)
        agg_exp_avg_sq = tree_map(lambda v, u: agg_beta2 * v + (1 - agg_beta2) * jnp.square(u), agg_exp_avg_sq, final_updates)
        final_updates = tree_map(lambda m, v: (m / bias_correction1) / (jnp.sqrt(v / bias_correction2) + eps), agg_exp_avg, agg_exp_avg_sq)

    final_updates = tree_map(lambda x: lr * x, final_updates)

    new_state = {
        'step': step,
        'exp_avg': [s[0] for s in new_states],
        'exp_avg_sq': [s[1] for s in new_states],
        'max_exp_avg_sq': [s[2] for s in new_states],
        'agg_exp_avg': agg_exp_avg,
        'agg_exp_avg_sq': agg_exp_avg_sq
    }

    return final_updates, new_state

def multiadam_optimizer(amsgrad=False, beta1=0.99, beta2=0.99, lr=1e-3, weight_decay=0, eps=1e-8, maximize=False, group_weights=None, agg_momentum=False, agg_beta1=0.0, agg_beta2=0.0):
    def init_state(params):
        n_groups = len(group_weights) if group_weights is not None else 1
        return init_multiadam_state(params, n_groups)

    def update_fn(grads, state, params):
        return multiadam_update(grads, state, params, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize, group_weights, agg_momentum, agg_beta1, agg_beta2)

    return optax.GradientTransformation(init_state, update_fn)