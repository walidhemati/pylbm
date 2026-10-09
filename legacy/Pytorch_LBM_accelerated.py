"""Original single-file PyTorch D2Q9 lid-driven cavity (pre-2026 version of pylbm).

Kept verbatim (apart from this docstring and the ``main()`` wrapper at the bottom, which
makes the module importable) as the "torch-compile / original" baseline of the
optimization ladder. Known physics issues, documented in the pylbm paper:

* +y points towards *decreasing* row index (``torch.roll`` shift -1 on rows for c_y=+1),
  so the "lid" at row -1 is physically the y=0 wall; the image is mirrored when plotted
  with ``origin="lower"``.
* The side walls swap 5<->6 and 8<->7, a specular (free-slip) reflection, not
  bounce-back; the no-slip condition is only imposed by zeroing ``u`` for the next
  equilibrium.
* The lid density uses ``f2+f5+f6+2*f4`` instead of ``f0+f1+f3+2*(f4+f7+f8)``.

Run it with ``python legacy/Pytorch_LBM_accelerated.py --n 512 --steps 2000``.
"""

from __future__ import annotations
import torch
import time
import matplotlib.pyplot as plt

# ────────────────────────── Configuration ────────────────────────────
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nx, Ny = 3000, 3000               # grid resolution (x, y) 
tau = 1.0/1.8                        # relaxation time 
u_lid = 0.05                     # lid velocity (lattice units)
max_iter = 100000                   # hard iteration cap
viz_every = 50                   # live‑plot refresh interval
rho0 = 1.0                      # initial density

# Visualization control
enable_plotting = False          # Set to False to disable all plotting for performance
show_final_plot = False          # Set to False to disable final streamlines plot


# ───────────────────────── Lattice constants ─────────────────────────
c = torch.tensor([[ 0, 0], [ 1, 0], [ 0, 1], [-1, 0], [ 0,-1],
                  [ 1, 1], [-1, 1], [-1,-1], [ 1,-1]], dtype=torch.float32, device=DEVICE)  # (9,2)
w = torch.tensor([4/9] + [1/9]*4 + [1/36]*4, dtype=torch.float32, device=DEVICE)            # (9,)
# Split components for dot‑products
cx, cy = c[:, 0], c[:, 1]

# Strategic torch.compile with explicit returns for proper updates
@torch.compile()
def collision_streaming_step(f, rho, u, cx, cy, w, inv_tau):
    """Compiled collision step with explicit return."""
    cu = u[..., 0, None] * cx + u[..., 1, None] * cy
    usq = (u**2).sum(-1, keepdim=True)
    feq = rho[..., None] * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    f = f + (feq - f) * inv_tau
    f_streamed = torch.empty_like(f)
    f_streamed[..., 0] = f[..., 0]
    f_streamed[..., 1] = torch.roll(f[..., 1], shifts=( 0,  1), dims=(0, 1))
    f_streamed[..., 2] = torch.roll(f[..., 2], shifts=(-1,  0), dims=(0, 1))
    f_streamed[..., 3] = torch.roll(f[..., 3], shifts=( 0, -1), dims=(0, 1))
    f_streamed[..., 4] = torch.roll(f[..., 4], shifts=( 1,  0), dims=(0, 1))
    f_streamed[..., 5] = torch.roll(f[..., 5], shifts=(-1,  1), dims=(0, 1))
    f_streamed[..., 6] = torch.roll(f[..., 6], shifts=(-1, -1), dims=(0, 1))
    f_streamed[..., 7] = torch.roll(f[..., 7], shifts=( 1, -1), dims=(0, 1))
    f_streamed[..., 8] = torch.roll(f[..., 8], shifts=( 1,  1), dims=(0, 1))
    return f_streamed


@torch.compile()
def compute_macroscopic(f, cx, cy):
    """Compiled macroscopic field computation without torch.sum for better performance."""
    # Manual density computation - faster than torch.sum
    rho = f[..., 0] + f[..., 1] + f[..., 2] + f[..., 3] + f[..., 4] + f[..., 5] + f[..., 6] + f[..., 7] + f[..., 8]
    
    u_x = (f[..., 1] * cx[1] + f[..., 2] * cx[2] + f[..., 3] * cx[3] + f[..., 4] * cx[4] + 
           f[..., 5] * cx[5] + f[..., 6] * cx[6] + f[..., 7] * cx[7] + f[..., 8] * cx[8]) / rho
    
    u_y = (f[..., 1] * cy[1] + f[..., 2] * cy[2] + f[..., 3] * cy[3] + f[..., 4] * cy[4] + 
           f[..., 5] * cy[5] + f[..., 6] * cy[6] + f[..., 7] * cy[7] + f[..., 8] * cy[8]) / rho
    
    # Manual velocity tensor construction - faster than torch.stack
    u = torch.empty((*f.shape[:-1], 2), device=f.device, dtype=f.dtype)
    u[..., 0] = u_x
    u[..., 1] = u_y
    
    # Enforce no-slip conditions (but NOT on the moving lid at top!)
    u[:, 0, :] = 0.0   # left wall
    u[:, -1, :] = 0.0  # right wall
    u[0, :, :] = 0.0   # bottom wall
    # Note: top wall (u[-1, :, :]) is NOT set to zero - it's the moving lid!
    
    return rho, u

@torch.compile()
def apply_boundary_conditions(f, u_lid_tensor, cx, cy, w):
    """Compiled boundary conditions using torch.compile friendly operations."""
    
    # Bounce-back boundaries using separate operations instead of simultaneous swaps
    # Left wall (x = 0): directions 1,5,8 bounce to 3,6,7
    temp_1 = f[:, 0, 1].clone()
    temp_5 = f[:, 0, 5].clone()
    temp_8 = f[:, 0, 8].clone()
    f[:, 0, 1] = f[:, 0, 3]
    f[:, 0, 5] = f[:, 0, 6]
    f[:, 0, 8] = f[:, 0, 7]
    f[:, 0, 3] = temp_1
    f[:, 0, 6] = temp_5
    f[:, 0, 7] = temp_8
    
    # Right wall (x = Nx-1): directions 3,6,7 bounce to 1,5,8
    temp_3 = f[:, -1, 3].clone()
    temp_6 = f[:, -1, 6].clone()
    temp_7 = f[:, -1, 7].clone()
    f[:, -1, 3] = f[:, -1, 1]
    f[:, -1, 6] = f[:, -1, 5]
    f[:, -1, 7] = f[:, -1, 8]
    f[:, -1, 1] = temp_3
    f[:, -1, 5] = temp_6
    f[:, -1, 8] = temp_7
    
    # Bottom wall (y = 0): directions 2,5,6 bounce to 4,7,8
    temp_2 = f[0, :, 2].clone()
    temp_5_bot = f[0, :, 5].clone()
    temp_6_bot = f[0, :, 6].clone()
    f[0, :, 2] = f[0, :, 4]
    f[0, :, 5] = f[0, :, 7]
    f[0, :, 6] = f[0, :, 8]
    f[0, :, 4] = temp_2
    f[0, :, 7] = temp_5_bot
    f[0, :, 8] = temp_6_bot
    
    # Moving lid boundary (top wall) - avoid .sum() and complex indexing
    # Manual sum instead of .sum(-1)
    rho_top = (f[-1, :, 0] + f[-1, :, 1] + f[-1, :, 3] + 
               f[-1, :, 2] + f[-1, :, 5] + f[-1, :, 6] + 
               2.0 * f[-1, :, 4])
    
    # Precompute lid velocity dot products
    cu_top_0 = u_lid_tensor[0] * cx[0] + u_lid_tensor[1] * cy[0]
    cu_top_1 = u_lid_tensor[0] * cx[1] + u_lid_tensor[1] * cy[1]
    cu_top_2 = u_lid_tensor[0] * cx[2] + u_lid_tensor[1] * cy[2]
    cu_top_3 = u_lid_tensor[0] * cx[3] + u_lid_tensor[1] * cy[3]
    cu_top_4 = u_lid_tensor[0] * cx[4] + u_lid_tensor[1] * cy[4]
    cu_top_5 = u_lid_tensor[0] * cx[5] + u_lid_tensor[1] * cy[5]
    cu_top_6 = u_lid_tensor[0] * cx[6] + u_lid_tensor[1] * cy[6]
    cu_top_7 = u_lid_tensor[0] * cx[7] + u_lid_tensor[1] * cy[7]
    cu_top_8 = u_lid_tensor[0] * cx[8] + u_lid_tensor[1] * cy[8]
    
    # Manual usq calculation instead of tensor operations
    usq_top = u_lid_tensor[0] * u_lid_tensor[0] + u_lid_tensor[1] * u_lid_tensor[1]
    
    # Calculate equilibrium distributions manually for each direction
    fe_top_0 = rho_top * w[0] * (1.0 + 3.0*cu_top_0 + 4.5*cu_top_0*cu_top_0 - 1.5*usq_top)
    fe_top_1 = rho_top * w[1] * (1.0 + 3.0*cu_top_1 + 4.5*cu_top_1*cu_top_1 - 1.5*usq_top)
    fe_top_2 = rho_top * w[2] * (1.0 + 3.0*cu_top_2 + 4.5*cu_top_2*cu_top_2 - 1.5*usq_top)
    fe_top_3 = rho_top * w[3] * (1.0 + 3.0*cu_top_3 + 4.5*cu_top_3*cu_top_3 - 1.5*usq_top)
    fe_top_4 = rho_top * w[4] * (1.0 + 3.0*cu_top_4 + 4.5*cu_top_4*cu_top_4 - 1.5*usq_top)
    fe_top_5 = rho_top * w[5] * (1.0 + 3.0*cu_top_5 + 4.5*cu_top_5*cu_top_5 - 1.5*usq_top)
    fe_top_6 = rho_top * w[6] * (1.0 + 3.0*cu_top_6 + 4.5*cu_top_6*cu_top_6 - 1.5*usq_top)
    fe_top_7 = rho_top * w[7] * (1.0 + 3.0*cu_top_7 + 4.5*cu_top_7*cu_top_7 - 1.5*usq_top)
    fe_top_8 = rho_top * w[8] * (1.0 + 3.0*cu_top_8 + 4.5*cu_top_8*cu_top_8 - 1.5*usq_top)
    
    # Apply moving lid boundary conditions
    f[-1, :, 2] = fe_top_2 + f[-1, :, 4] - fe_top_4
    f[-1, :, 5] = fe_top_5 + f[-1, :, 7] - fe_top_7
    f[-1, :, 6] = fe_top_6 + f[-1, :, 8] - fe_top_8
    
    return f

def lbm_step_strategic(f, f_temp, rho, u, u_lid_tensor, cx, cy, w, inv_tau):
    """Strategic compilation with explicit tensor updates."""
    # Step 1: Collision (compiled) - returns new f
    f = collision_streaming_step(f, rho, u, cx, cy, w, inv_tau)
     
    f= apply_boundary_conditions(f, u_lid_tensor, cx, cy, w)
    
    # Step 4: Macroscopic fields (compiled) - returns new rho, u
    rho, u = compute_macroscopic(f, cx, cy)
    
    return f, rho, u


def main() -> None:
    """Run the original benchmark loop (grid size and steps from the command line)."""
    import argparse
    global Nx, Ny, max_iter
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=Nx)
    ap.add_argument('--steps', type=int, default=max_iter)
    a = ap.parse_args()
    Nx = Ny = a.n
    max_iter = a.steps
    # ────────────────────── Field initialisation ─────────────────────────
    print("🎯 Strategic Compilation LBM Initialization...")
    rho = torch.full((Ny, Nx), rho0, device=DEVICE)
    u   = torch.zeros((Ny, Nx, 2), device=DEVICE)

    # Initial equilibrium distribution
    cu = u[..., 0, None] * cx + u[..., 1, None] * cy
    usq = (u**2).sum(-1, keepdim=True)
    f = rho[..., None] * w * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)

    # Pre-allocate memory-efficient buffers
    f_temp = torch.empty_like(f)
    u_lid_tensor = torch.tensor([u_lid, 0.0], device=DEVICE)

    # Only allocate visualization buffers if plotting is enabled
    if enable_plotting:
        speed_vis_cpu = torch.empty((Ny, Nx), dtype=torch.float32)  # CPU buffer for visualization

    # Precompute inverse tau for faster collision
    inv_tau = 1.0 / tau

    print("🚀 Starting Strategic Simulation...")

    # ───────────────────── Live visual initialisation ────────────────────
    if enable_plotting:
        plt.ion()
        fig_live, ax_live = plt.subplots(figsize=(6, 5))
        img = ax_live.imshow(torch.zeros((Ny, Nx)), origin="lower", cmap="turbo", vmin=0, vmax=u_lid*1.2)
        fig_live.colorbar(img, ax=ax_live, label="|u| (lattice units)")
        ax_live.set_title("Velocity magnitude – live")
        fig_live.tight_layout()
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

    # Warm up compiled functions
    print("⏳ Warming up compiled functions...")
    for _ in range(10):
        f, rho, u = lbm_step_strategic(f, f_temp, rho, u, u_lid_tensor, cx, cy, w, inv_tau)

    print("🏁 Running main simulation...")

    start_time = time.time()

    for iteration in range(max_iter):
        f, rho, u = lbm_step_strategic(f, f_temp, rho, u, u_lid_tensor, cx, cy, w, inv_tau)
        # Live visual every `viz_every` steps
        if enable_plotting and iteration % viz_every == 0:
            speed_magnitude = u.norm(dim=-1)
            speed_vis_cpu.copy_(speed_magnitude)
            speed_vis_cpu = speed_vis_cpu.detach().cpu()
            img.set_data(speed_vis_cpu)
            img.set_clim(vmin=0, vmax=speed_vis_cpu.max().item())
            fig_live.canvas.draw()
            fig_live.canvas.flush_events()

    end_time = time.time()

    # ────────────────────── Performance Analysis ─────────────────────────
    total_runtime = end_time - start_time
    final_million_lups = (max_iter * Nx * Ny) / (total_runtime * 1e6)


    print(f"\n🎯 STRATEGIC Performance Summary:")
    print(f"⚡ Total iterations: {max_iter}")
    print(f"🔥 Total lattice updates: {max_iter * Nx * Ny:,}")
    print(f"🚀 Average MLUPS: {final_million_lups:.3f}")
    print(f"⏱️  Total runtime: {total_runtime:.3f} seconds")


    # ───────────────── Refresh live figure one last time ─────────────────
    if enable_plotting:
        final_speed = u.norm(dim=-1).detach().cpu()
        img.set_data(final_speed)
        img.set_clim(vmin=0, vmax=final_speed.max().item())
        ax_live.set_title("Velocity magnitude – final")
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()

    # ───────────────────────── Final visualisations ──────────────────────
    if show_final_plot:
        plt.ioff()
        print("Plotting final snapshot with streamlines…")

        final_speed = u.norm(dim=-1).detach().cpu() if not enable_plotting else final_speed
        speed = final_speed.numpy()
        step = 1  # full resolution for accurate streamlines
        Y, X = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing="ij")
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        U_np = u[..., 0].cpu().numpy()
        V_np = -u[..., 1].cpu().numpy()  # Flip sign: array row index increases upwards, but positive uy is north (row-), so negate

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(speed, origin="lower", cmap="turbo")
        ax.streamplot(X_np, Y_np, U_np, V_np, density=2.0, linewidth=0.8, arrowsize=1)
        ax.set_title("Velocity magnitude + streamlines (steady state)")
        plt.tight_layout()
        plt.show()
    else:
        print("Final visualization disabled - calculation completed successfully")



if __name__ == "__main__":
    main()
