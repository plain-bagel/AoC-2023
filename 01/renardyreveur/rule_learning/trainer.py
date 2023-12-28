import pickle
from pathlib import Path

import jax
import jax.numpy as jnp

from training.data_generator import CHARS, CalibrationDataLoader
from training.init_params import initialize_params
from training.modelling.calibration_model import (
    calibration_model,
    CalibrationModelConfig,
)
from training.modelling.loss_fn import calibration_loss
from training.optimizer import adam

# --- PARAMETERS ---
MODEL_DIM = 32
NUM_HEADS = 2
NUM_LAYERS = 1
MAX_LENGTH = 60
OUTPUT_DIM = 10

# -- Training
ITERATIONS = 1000
BATCH_SIZE = 512
LR = 1e-3
WARMUP_STEPS = 50
# ------------------


def _prep_test_data(max_length: int) -> tuple[jnp.ndarray, list[int], list[int]]:
    """AoC Input data -> Not seen during training"""
    # Prepare test data
    test_data = Path("01/input.txt").read_text().strip().split("\n")
    test_data = [x.strip() for x in test_data]
    test_data = [x + " " * (max_length - len(x)) for x in test_data]
    targets = ["".join([y for y in x if y.isdigit()]) for x in test_data]

    test_data = jnp.array(
        [[CHARS.index(x) for x in test_data[i]] for i in range(len(test_data))]
    )
    first_num_targets = [int(x[0]) for x in targets]
    last_num_targets = [int(x[-1]) for x in targets]
    return test_data, first_num_targets, last_num_targets


def _test_model(
    data: jnp.ndarray,
    predicted: tuple[jnp.ndarray, jnp.ndarray],
    targets: tuple[list[int], list[int]],
) -> None:
    """Test the model against unseen data"""
    # Predictions
    first_num_pred, last_num_pred = predicted
    first_num_pred = jnp.argmax(first_num_pred, axis=1)
    last_num_pred = jnp.argmax(last_num_pred, axis=1)

    # Targets
    first_num_target, last_num_target = targets

    # Compute accuracy
    correct = sum(
        [
            1
            for i in range(len(data))
            if first_num_pred[i] == first_num_target[i]
            and last_num_pred[i] == last_num_target[i]
        ]
    )
    print(f"Test accuracy: {correct} / {len(data)}, ({correct*100 / len(data):.2f}%)")


def train(model_cfg: CalibrationModelConfig, batch_size: int, lr: float):
    """Train the model"""
    # Initialize model params
    params = initialize_params(model_cfg)
    model = jax.tree_util.Partial(calibration_model, cfg=model_cfg)
    print(
        f"Number of parameters: {sum(p.size for p in jax.tree_util.tree_flatten(params)[0])}"
    )

    # Create dataset
    dataloader = CalibrationDataLoader(batch_size=batch_size)

    # Create Adam optimizer
    opt_state = {
        "step": 0,
        "lr": lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "mu": jax.tree_util.tree_map(jnp.zeros_like, params),  # first moment
        "nu": jax.tree_util.tree_map(jnp.zeros_like, params),  # second moment
    }

    # Get test data
    _test_data, _fn_test_targets, _ln_test_targets = _prep_test_data(
        max_length=model_cfg.max_length
    )

    # Create loss function
    cal_loss = jax.tree_util.Partial(calibration_loss, model=model)

    # Step function
    # @jax.jit - unstable when turned on (it does converge eventually, but not to 100% test accuracy)
    def step(model_params, opt_params, batch_input, first_num_target, last_num_target):
        loss_value, grads = jax.value_and_grad(cal_loss)(
            model_params,
            x=batch_input,
            first_num_target=first_num_target,
            last_num_target=last_num_target,
        )
        # Clip gradients
        grads = jax.tree_map(lambda g: jnp.clip(g, -1, 1), grads)
        # Update model parameters with Adam
        model_params, opt_params = adam(grads, opt_params, model_params)
        return model_params, opt_state, loss_value

    # Train - Iterate over batches
    best_loss = 100
    for batch_idx, (_, x, y1, y2) in enumerate(iter(dataloader)):
        # --- Learning Rate Scheduling ---
        # warmup then cosine decay to 0 over final 1000 steps
        opt_state["lr"] = (
            (model_cfg.model_dim ** (-1.4) / WARMUP_STEPS) * (batch_idx + 1)
            if batch_idx < WARMUP_STEPS
            else model_cfg.model_dim ** (-1.4)
            * (
                1
                + jnp.cos(
                    jnp.pi * (batch_idx - WARMUP_STEPS) / (ITERATIONS - WARMUP_STEPS)
                )
            )
            / 2
        )

        # --- Compute single training step ---
        params, opt_state, loss = step(
            model_params=params,
            opt_params=opt_state,
            batch_input=x,
            first_num_target=y1,
            last_num_target=y2,
        )

        # --- Logging and Testing ---
        if batch_idx % 10 == 0:
            # Log loss
            print(
                f"\nIteration: {batch_idx}, Loss: {loss}}}, LR: {opt_state['lr']:.4f}"
            )

            # If loss is better than previous best, save the model
            if loss < best_loss:
                best_loss = loss
                print("Lowest loss so far, Saving model...")
                with open(
                    "01/renardyreveur/rule_learning/checkpoints/trained_model_params.pkl",
                    "wb",
                ) as f:
                    pickle.dump(params, f)

            # Test the model
            first_num_preds, last_num_preds = model(params, _test_data)
            _test_model(
                _test_data,
                (first_num_preds, last_num_preds),
                (_fn_test_targets, _ln_test_targets),
            )

        if batch_idx > ITERATIONS:
            break


if __name__ == "__main__":
    train(
        CalibrationModelConfig(
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            max_length=MAX_LENGTH,
            output_dim=OUTPUT_DIM,
            character_size=len(CHARS),
        ),
        BATCH_SIZE,
        LR,
    )
