"""Generate a random Q8.8 input tensor and save it in NumPy .npy format.

The output layout is:
    B x C x W x H

Each value is stored as a signed 16-bit integer in Q8.8 format, so the
integer value in the saved array represents the real value multiplied by 256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


Q_FRAC_BITS = 8
Q_SCALE = 1 << Q_FRAC_BITS
Q_MIN = -32768
Q_MAX = 32767


def generate_random_q8_8(
    batch_size: int,
    channels: int,
    width: int,
    height: int,
    seed: int | None = None,
) -> np.ndarray:
    """Return a random tensor with shape (B, C, W, H) as int16 Q8.8 values."""

    rng = np.random.default_rng(seed)
    data = rng.integers(
        low=Q_MIN,
        high=Q_MAX + 1,
        size=(batch_size, channels, width, height),
        dtype=np.int16,
    )
    return data


def load_input_shape_from_json(json_path: Path) -> tuple[int, int, int, int]:
    """Extract B, C, W, H from the layer whose type is input."""

    with json_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    layers = config.get("layers", [])
    for layer in layers:
        if layer.get("type") == "input":
            try:
                return (
                    int(layer["B"]),
                    int(layer["C"]),
                    int(layer["W"]),
                    int(layer["H"]),
                )
            except KeyError as exc:
                raise KeyError(
                    f"Input layer in {json_path} is missing field {exc.args[0]!r}"
                ) from exc

    raise ValueError(f"No layer with type 'input' found in {json_path}")


def save_q8_8_npy(data: np.ndarray, output_path: Path) -> None:
    """Save the tensor in NumPy .npy format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data.astype(np.int16, copy=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a random Q8.8 tensor and save it as a binary file."
    )
    parser.add_argument(
        "-j",
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Optional JSON file to read B, C, W, H from the input layer.",
    )
    parser.add_argument("-B", "--batch-size", type=int, default=1)
    parser.add_argument("-C", "--channels", type=int, default=3)
    parser.add_argument("-W", "--width", type=int, default=224)
    parser.add_argument("-H", "--height", type=int, default=224)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("Input_rand.npy"),
        help="Output .npy file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = args.batch_size
    channels = args.channels
    width = args.width
    height = args.height

    if args.json_path is not None:
        batch_size, channels, width, height = load_input_shape_from_json(args.json_path)

    data = generate_random_q8_8(
        batch_size=batch_size,
        channels=channels,
        width=width,
        height=height,
        seed=args.seed,
    )
    save_q8_8_npy(data, args.output)
    print(
        f"Saved {args.output} with shape "
        f"({batch_size}, {channels}, {width}, {height})"
    )


if __name__ == "__main__":
    main()
