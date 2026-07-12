"""Generate random Q8.8 weight tensors for each convolution layer in a JSON file.

For every layer whose type is "conv", this script creates a 4D tensor with
shape K x C x F1 x F2 and saves it as:

    Weight_1.npy
    Weight_2.npy
    ...

The layers are processed in the order they appear in the JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


Q_MIN = -32768
Q_MAX = 32767


def load_conv_layers(json_path: Path) -> list[dict]:
    """Return the layers whose type is conv, preserving JSON order."""

    with json_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    layers = config.get("layers", [])
    conv_layers = [layer for layer in layers if layer.get("type") == "conv"]
    if not conv_layers:
        raise ValueError(f"No layers with type 'conv' were found in {json_path}")
    return conv_layers


def generate_random_q8_8(shape: tuple[int, int, int, int], seed: int | None = None) -> np.ndarray:
    """Generate a random Q8.8 tensor stored as int16."""

    rng = np.random.default_rng(seed)
    return rng.integers(low=Q_MIN, high=Q_MAX + 1, size=shape, dtype=np.int16)


def save_npy(data: np.ndarray, output_path: Path) -> None:
    """Save the tensor to a NumPy .npy file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data.astype(np.int16, copy=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random Q8.8 weights for every conv layer in a JSON file."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the network JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where Weight_#.npy files will be written.",
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
    conv_layers = load_conv_layers(args.json_path)

    for index, layer in enumerate(conv_layers, start=1):
        try:
            k = int(layer["K"])
            c = int(layer["C"])
            f1 = int(layer["F1"])
            f2 = int(layer["F2"])
        except KeyError as exc:
            raise KeyError(
                f"Conv layer #{index} in {args.json_path} is missing field {exc.args[0]!r}"
            ) from exc

        weights = generate_random_q8_8((k, c, f1, f2), seed=args.seed)
        output_path = args.output_dir / f"Weight_{index}.npy"
        save_npy(weights, output_path)
        print(f"Saved {output_path} with shape ({k}, {c}, {f1}, {f2})")


if __name__ == "__main__":
    main()
