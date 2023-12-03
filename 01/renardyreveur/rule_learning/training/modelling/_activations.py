import jax.numpy as jnp


def softmax(in_x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Multidimensional softmax"""
    # Subtract max for numerical stability
    x_max = jnp.max(in_x, axis=axis, keepdims=True)
    unnormalized = jnp.exp(in_x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=axis, keepdims=True)


def relu(in_x: jnp.ndarray) -> jnp.ndarray:
    """Rectified Linear Unit activation function"""
    return jnp.maximum(in_x, 0)
