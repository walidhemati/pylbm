"""Render the README hero animation: the Karman vortex street of benchmark 2D-2.

    python figures/make_hero.py   # needs a CUDA GPU; writes docs/media/hero.gif (< 4 MB)

The flow is computed with the torch.compile gather backend (the fused kernels only handle
box geometries); colour is vorticity in units of U/D.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(d: int = 30, frames: int = 72, every: int | None = None) -> None:
    import matplotlib
    from PIL import Image

    from pylbm import backends as B
    from pylbm.validation.karman import schafer_turek_problem

    u_mean = 0.05
    p = schafer_turek_problem(d=d, u_mean=u_mean, collision="trt")
    s = B.make("torch-compile", "gather", p)
    t_conv = int(d / u_mean)
    s.step(40 * t_conv)  # develop the street
    every = every or max(2, int(2 * 3.4 * t_conv / frames))  # ~2 shedding periods
    cmap = matplotlib.colormaps["RdBu_r"]
    imgs = []
    for _ in range(frames):
        s.step(every)
        _, u = s.macroscopic()
        w = (np.gradient(u[1], axis=1) - np.gradient(u[0], axis=0)) * d / u_mean
        rgb = cmap(np.clip((w + 3.0) / 6.0, 0, 1))[..., :3]
        rgb[p.solid] = (0.15, 0.15, 0.15)
        img = Image.fromarray((255 * rgb[::-1]).astype(np.uint8))
        img = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
        imgs.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))
    out = ROOT / "docs" / "media" / "hero.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=50, loop=0,
                 optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB, {len(imgs)} frames)")


if __name__ == "__main__":
    main()
