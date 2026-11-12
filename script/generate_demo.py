"""Generate a small, reproducible Gaussian model and aligned reference volume."""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    centers = np.array([[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.25]])
    scales = np.array([[0.12, 0.3, 0.2], [0.15, 0.2, 0.32], [0.35, 0.06, 0.08]])
    rotations = np.repeat(np.eye(3)[None], len(centers), axis=0)
    density = np.array([0.8, 1.0, 0.6])
    np.savez_compressed(
        args.output / "gaussians.npz",
        centers=centers,
        scales=scales,
        rotations=rotations,
        density=density,
    )
    size = 64
    axis = -1 + (np.arange(size) + 0.5) * 2 / size
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    volume = np.zeros((size, size, size), dtype=np.float32)
    for center, scale, rotation, amplitude in zip(centers, scales, rotations, density):
        local = ((points - center) @ rotation) / scale
        q = np.sum(local**2, axis=-1)
        volume += np.where(q <= 9, amplitude * np.exp(-0.5 * q), 0)
    tifffile.imwrite(args.output / "reference.tiff", volume, photometric="minisblack")
    (args.output / "meta_data.json").write_text(
        json.dumps(
            {
                "bbox": [[-1, -1, -1], [1, 1, 1]],
                "scanner": {"nVoxel": [size, size, size]},
            },
            indent=2,
        )
    )
    print(f"Demo written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
