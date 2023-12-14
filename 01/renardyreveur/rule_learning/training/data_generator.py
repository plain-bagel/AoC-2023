import random
from typing import Sequence

import jax.numpy as jnp

CHARS = "abcdefghijklmnopqrstuvwxyz123456789 "
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def __sample_str_from_pop(
    population: Sequence[str], min_length: int, max_length: int
) -> str:
    return "".join(
        [
            random.sample(population, k=1)[0]
            for _ in range(random.randint(min_length, max_length))
        ]
    )


def generate_calibration_line(
    min_length: int = 2,
    max_length: int = 60,
    pad_to_max_length: bool = True,
) -> tuple[str, jnp.ndarray, tuple[int, int]]:
    """Generate a calibration line of random length"""

    # Sample a line length from between min_length and max_length
    line_length = random.randint(min_length, max_length)

    # For 10% of the cases - low number of digit characters, extreme positions, consecutive digits
    if random.random() < 0.1:
        # Choose number of consecutive digit characters
        consec_numbers = random.randint(1, 3)
        num_str = __sample_str_from_pop(
            "123456789", min_length=consec_numbers, max_length=consec_numbers
        )
        first_num, last_num = int(num_str[0]), int(num_str[-1])

        # Choose extreme position (Start, Middle, End) of full string
        position = random.randint(1, 3)
        if position == 1:
            calibration_line = num_str + __sample_str_from_pop(ALPHABET, 12, 50)
        elif position == 2:
            calibration_line = (
                __sample_str_from_pop(ALPHABET, 8, 25)
                + num_str
                + __sample_str_from_pop(ALPHABET, 8, 25)
            )
        else:
            calibration_line = __sample_str_from_pop(ALPHABET, 12, 50) + num_str

    # 90% of the time, just randomly generate a sequence of alphanumeric characters
    else:
        # Generate a line of random characters of length line_length
        first_num, last_num = 0, 0
        calibration_line = ""
        for i in range(line_length):
            char = random.choice(CHARS.strip())
            if char.isdigit():
                if first_num == 0:
                    first_num = int(char)
                last_num = int(char)
            calibration_line += char

    # Pad the line to max_length
    if pad_to_max_length:
        calibration_line += " " * (max_length - len(calibration_line))

    # Convert the line to an array of indices
    calibration_array = jnp.array([CHARS.index(x) for x in calibration_line])

    # Return the calibration line, array and the indices of the first and last numbers
    return calibration_line, calibration_array, (first_num, last_num)


# Infinite data loader with batch size
class CalibrationDataLoader:
    def __init__(self, batch_size: int = 512):
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            # Generate a batch of calibration lines
            batch_lines = []
            batch_arrays = []
            batch_first_nums = []
            batch_last_nums = []
            for _ in range(self.batch_size):
                line, array, (first_num, last_num) = generate_calibration_line()
                batch_lines.append(line)
                batch_arrays.append(array)
                batch_first_nums.append(first_num)
                batch_last_nums.append(last_num)

            # Convert the batch to an array
            batch_arrays = jnp.array(batch_arrays)
            batch_first_nums = jnp.array(batch_first_nums)
            batch_last_nums = jnp.array(batch_last_nums)

            # Yield the batch
            yield batch_lines, batch_arrays, batch_first_nums, batch_last_nums
