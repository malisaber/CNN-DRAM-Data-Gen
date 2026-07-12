"""Convert a NumPy weight tensor into a MATLAB-style .bin file.

This script takes two arguments:
    1. input_path  - path to the input .npy file
    2. output_path - path to the output .bin file

The input tensor must be 4D in K x C x F1 x F2 order. Values are written
using the same nested traversal style as the MATLAB reference code.

If the input array is floating-point, each value is multiplied by 256 and then
stored as a 16-bit word.
If the input array is already integer-based, values are written as 16-bit
words directly. Negative values are preserved using two's-complement bit
pattern, matching MATLAB-style uint16 conversion.
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
            f"Expected a 4D tensor in K x C x F1 x F2 order, got shape {tensor.shape}"
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
    """Write the tensor using the nested ordering expected by the MATLAB flow."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_float = np.issubdtype(tensor.dtype, np.floating)

    with output_path.open("wb") as f:
        k_size, c_size, f1_size, f2_size = tensor.shape
        for k in range(k_size):
            for c in range(c_size):
                for f2 in range(f2_size):
                    for f1 in range(f1_size):
                        word = to_uint16_word(tensor[k, c, f1, f2], scale_float)
                        f.write(np.array([word], dtype="<u2").tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a 4D NumPy weight tensor into a MATLAB-style .bin file."
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
