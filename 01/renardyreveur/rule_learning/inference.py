import pickle
from pathlib import Path

import jax.numpy as jnp
from jax.tree_util import Partial

from trainer import MODEL_DIM, NUM_HEADS, NUM_LAYERS, MAX_LENGTH, OUTPUT_DIM
from training.data_generator import CHARS
from training.modelling.calibration_model import (
    calibration_model,
    CalibrationModelConfig,
)


def main():
    # Load weights and biases
    with Path(
        "01/renardyreveur/rule_learning/checkpoints/trained_model_params.pkl"
    ).open("rb") as f:
        params = pickle.load(f)

    # Prepare model
    model_cfg = CalibrationModelConfig(
        model_dim=MODEL_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_length=MAX_LENGTH,
        output_dim=OUTPUT_DIM,
        character_size=len(CHARS),
    )
    model = Partial(calibration_model, params=params, cfg=model_cfg)

    # Parse input data
    with Path("01/input.txt").open("r") as f:
        test_data = f.readlines()
    test_data = [x.strip() for x in test_data]
    test_data = [x + " " * (60 - len(x)) for x in test_data]
    test_data = jnp.array(
        [[CHARS.index(x) for x in test_data[i]] for i in range(len(test_data))]
    )

    # Batched Inference - Single Forward Pass
    first_num_pred, last_num_pred = model(in_x=test_data)

    # Parse Calibrations
    first_num_pred = jnp.argmax(first_num_pred, axis=1)
    last_num_pred = jnp.argmax(last_num_pred, axis=1)
    calibration_values = [fi * 10 + la for fi, la in zip(first_num_pred, last_num_pred)]
    print(f"The sum of all calibration values are: {sum(calibration_values)}")


if __name__ == "__main__":
    main()
