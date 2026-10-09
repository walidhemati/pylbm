#!/usr/bin/env python3
"""
Lattice Boltzmann Method – Milestone 02 (Streaming Operator)
Python + PyTorch implementation that follows the specification exactly.

Assumptions & simplifications (per milestone):
* 2‑D, D2Q9 lattice
* Collision term set to zero (free streaming)
* Periodic boundary conditions on all sides
* Demonstration grid: 15 × 10 nodes (modifiable)

This script provides:
1. Data structures using PyTorch tensors (can run on CPU or GPU)
2. Functions to compute density and velocity fields
3. Streaming kernel using `torch.roll` (periodic BC)
4. Minimal visualization with `matplotlib`

Run the script directly or import its functions in a notebook.
"""

from __future__ import annotations
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

# -------------------------------------------------------------
# Configuration & constants
# -------------------------------------------------------------

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# D2Q9 discrete velocity set c_i (cx, cy)
C = torch.tensor([
    [ 0,  0],   # 0: rest
    [ 1,  0],   # 1: east  →
    [ 0,  1],   # 2: north ↑
    [-1,  0],   # 3: west  ←
    [ 0, -1],   # 4: south ↓
    [ 1,  1],   # 5: north‑east ↗
    [-1,  1],   # 6: north‑west ↖
    [-1, -1],   # 7: south‑west ↙
    [ 1, -1]    # 8: south‑east ↘
], dtype=torch.int64, device=DEVICE)

# -------------------------------------------------------------
# Core LBM helper functions
# -------------------------------------------------------------

def compute_density(f: torch.Tensor) -> torch.Tensor:
    """Return density ρ(x,y) = Σ_i f_i.

    Parameters
    ----------
    f : (9, NY, NX) tensor – distribution function.
    """
    return f.sum(dim=0)

def compute_velocity(f: torch.Tensor, c: torch.Tensor = C) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return velocity components (u_x, u_y).

    v(x,y) = (1/ρ) Σ_i f_i * c_i
    """
    rho = compute_density(f)
    # Avoid div‑by‑zero by replacing zeros with ones (velocity stays 0 there)
    rho_safe = torch.where(rho == 0, torch.ones_like(rho), rho)

    # Accumulate momentum
    u_x = torch.zeros_like(rho, dtype=f.dtype, device=f.device)
    u_y = torch.zeros_like(rho, dtype=f.dtype, device=f.device)
    for i in range(9):
        u_x += f[i] * c[i, 0].float()
        u_y += f[i] * c[i, 1].float()
    u_x /= rho_safe
    u_y /= rho_safe
    return u_x, u_y

def streaming(f: torch.Tensor, c: torch.Tensor = C) -> torch.Tensor:
    """Perform the streaming step with periodic boundaries.

    Each component f_i is shifted by one lattice link along its c_i.
    """
    streamed = torch.empty_like(f)
    # dim‑0: direction, dim‑1: y (rows), dim‑2: x (cols)
    for i in range(9):
        dx, dy = int(c[i, 0]), int(c[i, 1])
        # torch.roll positive shift moves data to larger indices (periodic)
        streamed[i] = torch.roll(f[i], shifts=(dy, dx), dims=(0, 1))
    return streamed

# -------------------------------------------------------------
# Demonstration / standalone run
# -------------------------------------------------------------

def main():  # noqa: D401 – simple runner
    # Lattice size (modifiable)
    NX, NY = 15, 10  # width, height
    NSTEPS = 50      # number of time iterations

    # Initialise distribution: uniform plus a pulse moving east
    rho0 = 1.0
    f = torch.full((9, NY, NX), rho0 / 9.0, dtype=DTYPE, device=DEVICE)

    # Add an east‑moving perturbation in the middle column
    mid_x = NX // 2
    f[1, :, mid_x] += 0.2  # boost east component (direction 1)

    # Time‑stepping loop (pure streaming)
    for _ in range(NSTEPS):
        f = streaming(f)

    # Diagnostics (move back to CPU for plotting)
    ux, uy = (t.cpu() for t in compute_velocity(f))
    rho = compute_density(f).cpu()

    # Visualise velocity field
    X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
    plt.figure(figsize=(6, 4))
    plt.title(f"Velocity field after {NSTEPS} steps")
    plt.quiver(X, Y, ux.numpy(), uy.numpy(), rho.numpy(), scale=1, scale_units="xy")
    plt.gca().invert_yaxis()  # (0,0) at top‑left like array indices
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
