import jax.numpy as jnp


def layer_norm(
    params: dict[str, jnp.ndarray],
    in_x: jnp.ndarray,
    dim: int = -1,
) -> jnp.ndarray:
    eps = 0.00001
    mean = jnp.mean(in_x, axis=dim, keepdims=True)
    var = jnp.var(in_x, axis=dim, keepdims=True)
    x = (in_x - mean) / jnp.sqrt(var + eps)
    new_shape = [1] * len(in_x.shape)
    new_shape[dim] = -1
    w = params["weight"].reshape(new_shape)
    b = params["bias"].reshape(new_shape)
    return jnp.multiply(x, w) + b
