import jax
import jax.numpy as jnp

from .modelling.calibration_model import CalibrationModelConfig


def lecun_normal(
    key: jnp.ndarray,
    shape: tuple[int, int],
) -> jnp.ndarray:
    """LeCun normal initialization"""
    stddev = jnp.sqrt(1.0 / shape[0]) / jnp.array(0.87962566103423978)
    return jax.random.truncated_normal(key, -2, 2, shape) * stddev


def initialize_params(
    cfg: CalibrationModelConfig,
) -> dict[str, jnp.ndarray]:
    """Initialize weights and biases for the model"""
    params = dict()

    # Embedding weights
    prng_key = jax.random.PRNGKey(0)
    new_key, char_emb_key, pos_emb_key = jax.random.split(prng_key, 3)
    params["char_emb"] = jnp.sqrt(1.0 / cfg.character_size) * jax.random.normal(
        key=char_emb_key, shape=(cfg.character_size, cfg.model_dim)
    )
    params["pos_emb"] = jnp.sqrt(1.0 / cfg.max_length) * jax.random.normal(
        key=pos_emb_key, shape=(cfg.max_length, cfg.model_dim)
    )

    # Attention weights
    for i in range(cfg.num_layers):
        # Attention qkv and concat weights
        new_key, qkv_key, concat_key, l1_key, l2_key = jax.random.split(new_key, 5)
        params[f"transformer_enc_{i}"] = {
            "attention": {
                "query": {
                    "weight": lecun_normal(
                        qkv_key, (cfg.model_dim, cfg.model_dim)
                    ).reshape(
                        (cfg.model_dim, cfg.num_heads, cfg.model_dim // cfg.num_heads)
                    ),
                    "bias": jnp.zeros(
                        (cfg.num_heads, cfg.model_dim // cfg.num_heads),
                        dtype=jnp.float32,
                    ),
                },
                "key": {
                    "weight": lecun_normal(
                        qkv_key, (cfg.model_dim, cfg.model_dim)
                    ).reshape(
                        (cfg.model_dim, cfg.num_heads, cfg.model_dim // cfg.num_heads)
                    ),
                    "bias": jnp.zeros(
                        (cfg.num_heads, cfg.model_dim // cfg.num_heads),
                        dtype=jnp.float32,
                    ),
                },
                "value": {
                    "weight": lecun_normal(
                        qkv_key, (cfg.model_dim, cfg.model_dim)
                    ).reshape(
                        (cfg.model_dim, cfg.num_heads, cfg.model_dim // cfg.num_heads)
                    ),
                    "bias": jnp.zeros(
                        (cfg.num_heads, cfg.model_dim // cfg.num_heads),
                        dtype=jnp.float32,
                    ),
                },
                "concat_weights": lecun_normal(
                    concat_key, (cfg.model_dim, cfg.model_dim)
                ),
                "concat_bias": jnp.zeros((cfg.model_dim,), dtype=jnp.float32),
            },
            "pre_ln_1": {
                "weight": jnp.ones((cfg.model_dim,), dtype=jnp.float32),
                "bias": jnp.zeros((cfg.model_dim,), dtype=jnp.float32),
            },
            "pre_ln_2": {
                "weight": jnp.ones((cfg.model_dim,), dtype=jnp.float32),
                "bias": jnp.zeros((cfg.model_dim,), dtype=jnp.float32),
            },
            "linear_1": {
                "weight": lecun_normal(l1_key, (cfg.model_dim, cfg.model_dim * 2)),
                "bias": jnp.zeros((cfg.model_dim * 2,), dtype=jnp.float32),
            },
            "linear_2": {
                "weight": lecun_normal(l2_key, (cfg.model_dim * 2, cfg.model_dim)),
                "bias": jnp.zeros((cfg.model_dim,), dtype=jnp.float32),
            },
        }

    # Concat projection weights
    new_key, concat_proj_key, concat_bias_key = jax.random.split(new_key, 3)
    params["concat_proj"] = {
        "weight": lecun_normal(
            concat_proj_key, (cfg.model_dim * cfg.max_length, cfg.model_dim)
        ),
        "bias": jnp.zeros((cfg.model_dim,), dtype=jnp.float32),
    }

    # Output layer weights
    (
        new_key,
        output1_key,
        output1_bias_key,
        output2_key,
        output2_bias_key,
    ) = jax.random.split(new_key, 5)
    params["output_first_num_proj"] = {
        "weight": lecun_normal(output1_key, (cfg.model_dim, cfg.output_dim)),
        "bias": jnp.zeros((cfg.output_dim,)),
    }
    params["output_second_num_proj"] = {
        "weight": lecun_normal(output2_key, (cfg.model_dim, cfg.output_dim)),
        "bias": jnp.zeros((cfg.output_dim,)),
    }

    return params
