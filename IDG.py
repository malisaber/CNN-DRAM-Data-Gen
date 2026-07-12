"""Convert a NumPy tensor into a MATLAB-style .bin file.
 
This script takes two arguments:
    1. input_path  - path to the input .npy file
    2. output_path - path to the output .bin file

The tensor must be 4D in B x C x W x H order. Values are written using the
same nested traversal as the MATLAB reference:

    for b in B
        for c in C
            for w in W
                for h in H
                    write value

If the input array is floating-point, each value is multiplied by 256 and then
stored as a 16-bit word.
If the input array is already integer-based, values are written as 16-bit
words directly. Negative values are preserved using two's-complement bit
pattern, which matches MATLAB's `uint16(...)` conversion behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_tensor(input_path: Path) -> np.ndarray:
    """Load the .npy tensor and validate that it is 4D."""

    tensor = np.load(input_path)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D tensor in B x C x W x H order, got shape {tensor.shape}"
        )
    return tensor


def to_uint16_word(value: np.ndarray | int | float, scale_float: bool) -> np.uint16:
    """Convert one tensor element into the 16-bit word written to the file."""

    if scale_float:
        word = np.int16(np.rint(float(value) * 256.0))
    else:
        word = np.int16(value)

    return np.uint16(word)


def write_bin_like_matlab(tensor: np.ndarray, output_path: Path) -> None:
    """Write the tensor in the same nested-loop order as the MATLAB script."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_float = np.issubdtype(tensor.dtype, np.floating)

    with output_path.open("wb") as f:
        batch_size, channels, width, height = tensor.shape
        for b in range(batch_size):
            for c in range(channels):
                for w in range(width):
                    for h in range(height):
                        word = to_uint16_word(tensor[b, c, w, h], scale_float)
                        f.write(np.array([word], dtype="<u2").tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 4D NumPy tensor into a MATLAB-style .bin file."
    )
    parser.add_argument("input_path", type=Path, help="Path to the input .npy file.")
    parser.add_argument(
        "output_path", type=Path, help="Path to the output .bin file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tensor = load_tensor(args.input_path)
    write_bin_like_matlab(tensor, args.output_path)
    print(f"Saved {args.output_path} from {args.input_path} with shape {tensor.shape}")


if __name__ == "__main__":
    main()
