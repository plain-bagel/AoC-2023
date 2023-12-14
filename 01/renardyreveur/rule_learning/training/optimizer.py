from typing import Any

import jax
import jax.numpy as jnp


def adam(
    grads: jnp.ndarray,
    opt_params: dict[str, Any],
    params: dict[str, dict | jnp.ndarray],
) -> tuple[dict[str, dict | jnp.ndarray], dict[str, Any]]:
    # Momentum
    # Gradient direction is smoothed by exponentially weighing the moving averages
    opt_params["mu"] = jax.tree_map(
        lambda m, g: m * opt_params["betas"][0] + g * (1 - opt_params["betas"][0]),
        opt_params["mu"],
        grads,
    )

    # RMSProp
    # Gradient magnitude is smoothed such that it slows down near flats, and doesn't flick off at suboptimal gradients
    opt_params["nu"] = jax.tree_map(
        lambda v, g: v * opt_params["betas"][1] + g**2 * (1 - opt_params["betas"][1]),
        opt_params["nu"],
        grads,
    )

    # Increase step
    opt_params["step"] += 1

    # Estimation bias correction
    mu_hat = jax.tree_map(
        lambda m: m / (1 - opt_params["betas"][0] ** opt_params["step"]),
        opt_params["mu"],
    )
    nu_hat = jax.tree_map(
        lambda v: v / (1 - opt_params["betas"][1] ** opt_params["step"]),
        opt_params["nu"],
    )

    # Calculate update
    updates = jax.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + opt_params["eps"]), mu_hat, nu_hat
    )

    # Apply updates
    params = jax.tree_map(lambda w, dw: w - opt_params["lr"] * dw, params, updates)
    return params, opt_params
