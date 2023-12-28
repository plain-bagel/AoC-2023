from dataclasses import dataclass

import jax.numpy as jnp

from ._base_layers import linear, transformer_encoder


@dataclass
class CalibrationModelConfig:
    """Configuration for the calibration model"""

    model_dim: int
    num_heads: int
    num_layers: int
    max_length: int
    output_dim: int
    character_size: int


def calibration_model(
    params: dict[str, dict | jnp.ndarray],
    in_x: jnp.ndarray,
    cfg: CalibrationModelConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calibration model"""
    # Learned character embeddings
    x = jnp.take(params["char_emb"], in_x, axis=0)

    # Apply learned positional encoding
    pos_emb = jnp.take(params["pos_emb"], jnp.arange(cfg.max_length), axis=0)
    x = x + pos_emb

    # Transformer encoder
    for i in range(cfg.num_layers):
        x = transformer_encoder(params[f"transformer_enc_{i}"], x)

    # Concatenate over the sequence dimension
    x = x.reshape(-1, x.shape[-1] * x.shape[-2])
    x = linear(params["concat_proj"], x)

    # Project to first and last number categories
    first_num = linear(params["output_first_num_proj"], x)
    last_num = linear(params["output_second_num_proj"], x)
    return first_num, last_num
