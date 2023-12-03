from typing import Callable

import jax
import jax.numpy as jnp


def softmax_int(logits, labels):
    """Impl. from optax"""
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=-1))
    return log_normalizers - label_logits


def calibration_loss(
    params: dict[str, dict | jnp.ndarray],
    model: Callable,
    x: jnp.ndarray,
    first_num_target: jnp.ndarray,
    last_num_target: jnp.ndarray,
) -> jnp.ndarray:
    """Cross Entropy loss for a batch of examples, 'mean' reduction"""
    first_num_pred, last_num_pred = model(params, x)
    loss1 = softmax_int(first_num_pred, first_num_target)
    loss2 = softmax_int(last_num_pred, last_num_target)
    return jnp.mean(loss1) + jnp.mean(loss2)
