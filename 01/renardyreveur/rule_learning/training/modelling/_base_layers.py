import jax.numpy as jnp

from ._activations import relu, softmax
from ._normalization import layer_norm


def linear(
    params: dict[str, jnp.array],
    in_x: jnp.ndarray,
) -> jnp.ndarray:
    """Linear layer"""
    return jnp.dot(in_x, params["weight"]) + params["bias"]


def attention(
    params: dict[str, jnp.ndarray],
    in_x: jnp.ndarray,
) -> jnp.ndarray:
    """Attention layer"""
    # Project input to qkv space
    query = (
        jnp.einsum("bse,ehd->bshd", in_x, params["query"]["weight"])
        + params["query"]["bias"]
    )
    key = (
        jnp.einsum("bse,ehd->bshd", in_x, params["key"]["weight"])
        + params["key"]["bias"]
    )
    value = (
        jnp.einsum("bse,ehd->bshd", in_x, params["value"]["weight"])
        + params["value"]["bias"]
    )

    # Split qkv into q, k, v
    b, s, h, dim = query.shape
    query = query / jnp.sqrt(dim).astype(jnp.float32)

    # Compute attention weights
    attention_scores = jnp.einsum("bqhd,bkhd->bhqk", query, key)

    # Softmax
    attention_scores = softmax(attention_scores, axis=-1)

    # Compute attention output
    scaled_value = jnp.einsum("bhqk,bkhd->bqhd", attention_scores, value)

    # Concatenate heads
    scaled_value = scaled_value.reshape(b, s, h * dim)
    output = jnp.dot(scaled_value, params["concat_weights"]) + params["concat_bias"]
    return output


def transformer_encoder(
    params: dict[str, dict | jnp.ndarray],
    in_x: jnp.ndarray,
) -> jnp.ndarray:
    """Self-Attention + Feed Forward layer w/ residual connections and LayerNorm"""
    # Attention layer
    x = layer_norm(params["pre_ln_1"], in_x)  # Pre-LN
    x = attention(params["attention"], x)  # Self-Attention
    x = x + in_x  # Residual connection

    # Feed forward layer
    y = layer_norm(params["pre_ln_2"], x)  # Pre-LN
    y = linear(params["linear_1"], y)  # Linear expansive projection
    y = relu(y)  # ReLU activation
    y = linear(params["linear_2"], y)  # Linear compressive projection
    return y + x
