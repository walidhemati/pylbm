#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# D2Q9 lattice-Boltzmann (BGK) demo in PyTorch
#
# - streams → computes ρ,u → collisions
# - runs on CPU or GPU (CUDA) depending on availability
# - visualises density and |u| after a chosen number of steps
#
# 2025-06-18

import torch
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1.  Numerical parameters and helper tensors
# ----------------------------------------------------------------------
Nx, Ny   = 300, 300         # lattice size
nsteps   = 200              # number of time steps
tau      = 0.60             # relaxation time  (0 < tau, tau ≠ 0.5)
omega    = 1.0 / tau        # relaxation parameter ω = 1/τ
dtype    = torch.float32
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# D2Q9 velocity set ----------------------------------------------------
# integer copy for streaming (torch.roll needs ints)
c_int = torch.tensor(
    [[ 0,  0],           # 0  rest
     [ 1,  0], [ 0,  1], # 1,2  axis directions
     [-1,  0], [ 0, -1], # 3,4
     [ 1,  1], [-1,  1], # 5,6  diagonals
     [-1, -1], [ 1, -1]],# 7,8
    dtype=torch.int64,
    device=device)

# weights (same order as c_int)
w = torch.tensor(
    [4/9]             +          # rest particle
    4 * [1/9]         +          # axis directions
    4 * [1/36],                   # diagonals
    dtype=dtype,
    device=device)

# floating-point version for dot products (shape 2×9)
c_xy = c_int.to(dtype).t().contiguous()    # (2,Q) == (x/y, 9)

# integer shifts, one per population, for torch.roll
shifts = [(int(cx), int(cy)) for cx, cy in c_int.tolist()]

# reshape helpers for broadcasting
w_ = w.view(1, 1, 9)                       # (1,1,Q)


# ----------------------------------------------------------------------
# 2.  Equilibrium distribution function  f_eq(ρ,u)
# ----------------------------------------------------------------------
def feq(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    rho : (Nx,Ny)
    u   : (Nx,Ny,2)      – u[...,0]=u_x, u[...,1]=u_y
    returns
    feq : (Nx,Ny,Q)
    """
    cu  = torch.einsum('xyc,cq->xyq', u, c_xy)            # u·c_i
    usq = (u**2).sum(-1, keepdim=True)                    # |u|²

    # lattice-Boltzmann second-order Hermite expansion
    feq = rho.unsqueeze(-1) * w_ * (
            1.0 + 3.0*cu + 4.5*cu**2 - 1.5*usq)
    return feq


# ----------------------------------------------------------------------
# 3.  Initial condition ρ(x,y), u(x,y)
# ----------------------------------------------------------------------
# centred Gaussian bump in density
x = torch.arange(Nx, device=device)
y = torch.arange(Ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')
R2   = (X - Nx/2)**2 + (Y - Ny/2)**2
rho0 = 1.0 + 0.1*torch.exp(-R2 / (2*(Nx/10)**2))
rho0 = rho0.to(dtype)

u0   = torch.zeros(Nx, Ny, 2, dtype=dtype, device=device)  # quiescent

# initial discrete populations
f = feq(rho0, u0).clone()   # (Nx,Ny,Q)


# ----------------------------------------------------------------------
# 4.  Time integration loop
# ----------------------------------------------------------------------
for step in range(nsteps):
    # -------- 4.1 streaming (shift each population along its lattice vector)
    for q, (cx, cy) in enumerate(shifts):
        f[:, :, q] = torch.roll(f[:, :, q], shifts=(cx, cy), dims=(0, 1))

    # -------- 4.2 macroscopic fields ρ, u
    rho = f.sum(dim=2)                                          # (Nx,Ny)
    u   = torch.einsum('xyq,cq->xyc', f, c_xy) / rho.unsqueeze(-1)

    # -------- 4.3 collision step (BGK relaxation)
    f += omega * (feq(rho, u) - f)


# ----------------------------------------------------------------------
# 5.  Visualisation
# ----------------------------------------------------------------------
rho_cpu  = rho.cpu().numpy()
umag_cpu = torch.linalg.vector_norm(u, dim=2).cpu().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax1.imshow(rho_cpu, origin='lower', cmap='viridis')
ax1.set_title(f'Density after {nsteps} steps')
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(umag_cpu, origin='lower', cmap='plasma')
ax2.set_title('Velocity magnitude |u|')
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
