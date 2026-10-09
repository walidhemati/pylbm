import torch
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------- device  --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

# ------------------------------------------------ lattice & parameters  --
Nx, Ny = 800, 800           # lattice size (periodic BCs) - larger for better accuracy
nsteps = 100000              # timesteps for simulation - 100k steps for better graphs
save_every = 200             # store fields every N steps for visualization
eps = 0.1                  # shear-wave amplitude - reduced for stability over longer runs
ky = 2 * math.pi / Ny        # wave-number 2π/λ with one period along y

# Multiple omega values to test (keep 0 < ω < 2 as per notes)
omegas = [0.4, 0.6, 0.8, 1.0]

# ---------------------------- D2Q9 constants (vectors, weights, helpers) --
c_int = torch.tensor(
    [[ 0,  0], [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
     [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]], 
    dtype=torch.int64, device=device)

w = torch.tensor([4/9] + 4*[1/9] + 4*[1/36], dtype=dtype, device=device)
c_xy = c_int.to(dtype).t().contiguous()  # (2,9)
shifts = [(int(cx), int(cy)) for cx, cy in c_int.tolist()]
w_ = w.view(1, 1, 9)

# ---------------------------------------------------- equilibrium f_eq() --
def feq(rho, u):
    """
    Equilibrium distribution function for D2Q9 lattice
    f_i^eq(ρ,u) = w_i * ρ * [1 + 3*c_i·u + 9/2*(c_i·u)² - 3/2*u²]
    """
    cu = torch.einsum('xyc,cq->xyq', u, c_xy)
    usq = (u**2).sum(-1, keepdim=True)
    return rho.unsqueeze(-1) * w_ * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)

# ----------------------------------- one simulation → ν_measured, frames --
@torch.no_grad()
def run_shear_wave(omega):
    """
    Run a single shear-wave decay simulation for given omega
    
    Tasks from milestone:
    - Choose initial distribution ρ(r) = 1 and u_x(r) = ε*sin(2πy/n_y)
    - Observe dynamics and long-time behavior
    - Calculate kinematic viscosity from decay rate
    """
    tau = 1.0 / omega
    
    # ------------------------ initial macroscopic fields -----------------
    rho0 = torch.ones(Nx, Ny, dtype=dtype, device=device)
    
    # Initialize velocity with sinusoidal u_x profile
    y = torch.arange(Ny, device=device, dtype=dtype)
    u0 = torch.zeros(Nx, Ny, 2, dtype=dtype, device=device)
    u0[:, :, 0] = eps * torch.sin(ky * y).unsqueeze(0)  # u_x = ε*sin(2πy/Ny)
    
    # Initialize distribution functions to equilibrium
    f = feq(rho0, u0).clone()
    
    # Storage for diagnostics and visualization
    amplitude_history = []
    time_history = []
    frames_rho = []
    frames_u = []
    
    print(f"Running simulation for ω = {omega:.2f} (τ = {tau:.3f})")
    t_start = time.time()
    
    for step in range(nsteps + 1):
        # ---------- streaming step (advection) ---------------------------
        for q, (cx, cy) in enumerate(shifts):
            f[:, :, q] = torch.roll(f[:, :, q], shifts=(cx, cy), dims=(0, 1))
        
        # ---------- compute macroscopic fields ---------------------------
        rho = f.sum(2)  # density
        u = torch.einsum('xyq,cq->xyc', f, c_xy) / rho.unsqueeze(-1)  # velocity
        
        # ---------- BGK collision step -----------------------------------
        f += omega * (feq(rho, u) - f)
        
        # ---------- diagnostics and data collection ----------------------
        if step % save_every == 0:
            frames_rho.append(rho.cpu().numpy())
            frames_u.append(torch.linalg.vector_norm(u, 2, 2).cpu().numpy())
        
        # Progress reporting
        if step % (nsteps // 10) == 0 and step > 0:
            progress = step / nsteps * 100
            print(f"    Progress: {progress:.0f}%")
        
        amplitude = u[:, :, 0].abs().max().item()   # now A(0) == eps

        amplitude_history.append(amplitude)
        time_history.append(step)
    
    simulation_time = time.time() - t_start
    
    

    start_fit = len(time_history) // 10  
    end_fit = int(0.8 * len(time_history))
    
    log_amplitude = np.log(np.abs(amplitude_history[start_fit:end_fit]))
    times_fit = np.array(time_history[start_fit:end_fit])
    
    # Linear regression to find slope
    slope, intercept, r_value, p_value, std_err = linregress(times_fit, log_amplitude)
    
    # Calculate measured viscosity
    nu_measured = -slope / (ky**2)
    
    # Theoretical viscosity for BGK model: ν = (1/3) * (1/ω - 0.5)
    nu_theoretical = (1/3) * (1/omega - 0.5)
    
    print(f"  Simulation completed in {simulation_time:.1f}s")
    print(f"  Measured viscosity: ν = {nu_measured:.6f}")
    print(f"  Theoretical viscosity: ν = {nu_theoretical:.6f}")
    print(f"  Relative error: {abs(nu_measured - nu_theoretical)/nu_theoretical*100:.2f}%")
    print(f"  Fit quality (R²): {r_value**2:.4f}")
    print(f"  Fit range: steps {start_fit} to {end_fit} (of {len(time_history)})")
    
    return {
        'nu_measured': nu_measured,
        'nu_theoretical': nu_theoretical,
        'amplitude': amplitude_history,
        'time': time_history,
        'frames_rho': frames_rho,
        'frames_u': frames_u,
        'omega': omega,
        'fit_quality': r_value**2,
        'fit_range': (start_fit, end_fit),
        'slope': slope,
        'intercept': intercept
    }

# --------------------------------------------------------- plotting functions --
def plot_amplitude_decay(results, save_plot=True, t_max=90_000):
    """
    Plot shear‑wave amplitude decay on a *linear* y‑axis
    with a dark background and "k"‑style tick labels
    (0 k, 20 k, … t_max).

    Parameters
    ----------
    results : list of dict
        Output dictionaries from run_shear_wave (one per ω).
    save_plot : bool, default True
        If True, saves PNG called 'shear_wave_decay_dark.png'.
    t_max : int, default 90_000
        Upper bound of x‑axis in lattice‑time units.
    """
    # ----------------------------  style  ---------------------------------
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelcolor": "white",
        "xtick.color":  "0.8",
        "ytick.color":  "0.8",
        "grid.color":   "0.25",
    })

    # Consistent colour / line‑style palette
    colours  = ["#2df", "#29f", "#c0f", "#f06", "#fa0", "#7f7", "#ff9"]
    linestyl = ["dotted", "dashdot", "solid", "dashed", (0, (5, 2)),
                (0, (3, 1, 1, 1)), (0, (1, 1))]

    # ----------------------------  plot  -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, result in enumerate(results):
        omega = result["omega"]
        t     = np.array(result["time"])
        A     = np.abs(result["amplitude"])

        # Trim to t_max if simulation ran longer
        mask  = t <= t_max
        t, A  = t[mask], A[mask]

        ax.plot(t, A,
                color=colours[i % len(colours)],
                linestyle=linestyl[i % len(linestyl)],
                linewidth=2,
                label=rf"$\omega={omega}$")

    # ----------------------------  axes  ------------------------------------
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 0.10)                      # linear y‑axis → curved decay

    ax.set_xlabel(r"Time $\left(\frac{a}{c}\right)$", labelpad=6)
    ax.set_ylabel(r"Velocity Amplitude $\|\vec{u}_{max}\|$ $(c)$", labelpad=6)
    ax.set_title("Shear‑wave Decay for Different Relaxation Parameters", pad=10)

    # “k”‑style ticks: 0 k, 20 k, …
    ax.set_xticks(np.arange(0, t_max + 1, 20_000))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x else "0"))

    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    # ----------------------------  save/show  -------------------------------
    if save_plot:
        fname = "shear_wave_decay_dark.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        print(f"Amplitude‑decay plot saved as '{fname}'")

    plt.show()

def plot_viscosity_comparison(results, save_plot=True):
    """
    Plot measured vs theoretical viscosity
    """
    omegas = [result['omega'] for result in results]
    nu_measured = [result['nu_measured'] for result in results]
    nu_theoretical = [result['nu_theoretical'] for result in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(omegas, nu_measured, 'o-', label='Measured', markersize=8, linewidth=2)
    plt.plot(omegas, nu_theoretical, 's-', label='Theoretical', markersize=8, linewidth=2)
    
    plt.xlabel('Relaxation Parameter ω', fontsize=12)
    plt.ylabel('Kinematic Viscosity ν', fontsize=12)
    plt.title('Viscosity Measurement vs Theory', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('viscosity_comparison.png', dpi=150, bbox_inches='tight')
        print("Viscosity comparison plot saved as 'viscosity_comparison.png'")
    
    plt.show()

def plot_field_evolution(result, save_plot=True):
    """
    Plot evolution of density and velocity fields
    """
    frames_rho = result['frames_rho']
    frames_u = result['frames_u']
    omega = result['omega']
    
    # Select few representative time frames
    n_frames = len(frames_rho)
    selected_frames = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]
    
    fig, axes = plt.subplots(2, len(selected_frames), figsize=(15, 6))
    
    for i, frame_idx in enumerate(selected_frames):
        time_step = frame_idx * save_every
        
        # Density plot
        im1 = axes[0, i].imshow(frames_rho[frame_idx], origin='lower', cmap='viridis')
        axes[0, i].set_title(f't = {time_step}')
        axes[0, i].set_xlabel('x')
        if i == 0:
            axes[0, i].set_ylabel('Density ρ')
        
        # Velocity magnitude plot
        im2 = axes[1, i].imshow(frames_u[frame_idx], origin='lower', cmap='plasma')
        axes[1, i].set_xlabel('x')
        if i == 0:
            axes[1, i].set_ylabel('Velocity |u|')
    
    plt.suptitle(f'Field Evolution for ω = {omega:.1f}', fontsize=16)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'field_evolution_omega_{omega:.1f}.png', dpi=150, bbox_inches='tight')
        print(f"Field evolution plot saved as 'field_evolution_omega_{omega:.1f}.png'")
    
    plt.show()

def plot_exponential_fit(result, save_plot=True):
    """
    Plot the exponential decay fit for debugging
    """
    omega = result['omega']
    time = np.array(result['time'])
    amplitude = np.abs(result['amplitude'])
    fit_range = result['fit_range']
    slope = result['slope']
    intercept = result['intercept']
    
    plt.figure(figsize=(10, 6))
    
    # Plot full amplitude history
    plt.semilogy(time, amplitude, 'b-', alpha=0.7, label='Full amplitude')
    
    # Plot fitting range
    start_fit, end_fit = fit_range
    plt.semilogy(time[start_fit:end_fit], amplitude[start_fit:end_fit], 
                'r-', linewidth=2, label='Fit range')
    
    # Plot fitted line
    fit_line = np.exp(intercept + slope * time[start_fit:end_fit])
    plt.semilogy(time[start_fit:end_fit], fit_line, 'g--', linewidth=2, 
                label=f'Exponential fit (slope={slope:.6f})')
    
    plt.xlabel('Time (lattice units)', fontsize=12)
    plt.ylabel('Velocity Amplitude |A(t)|', fontsize=12)
    plt.title(f'Exponential Decay Fit for ω = {omega:.1f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f'exponential_fit_omega_{omega:.1f}.png', dpi=150, bbox_inches='tight')
        print(f"Exponential fit plot saved as 'exponential_fit_omega_{omega:.1f}.png'")
    
    plt.show()

# --------------------------------------------------------- main execution --
def main():
    """
    Main execution function following milestone requirements
    """
    print("="*60)
    print("MILESTONE 04: Shear-wave Decay & Viscosity Measurement")
    print("="*60)
    print(f"Lattice size: {Nx} × {Ny}")
    print(f"Time steps: {nsteps}")
    print(f"Initial amplitude: ε = {eps}")
    print(f"Omega values: {omegas}")
    print(f"Device: {device}")
    print("="*60)
    
    # Run simulations for all omega values
    results = []
    for omega in omegas:
        result = run_shear_wave(omega)
        results.append(result)
        print("-" * 40)
    
    # Generate plots as required by milestone
    print("\nGenerating plots...")
    plot_amplitude_decay(results, save_plot=True)
    plot_viscosity_comparison(results, save_plot=True)
    
    # Show field evolution for one representative case
    middle_idx = len(results) // 2
    plot_field_evolution(results[middle_idx], save_plot=True)
    
    # Show exponential fit for debugging (first omega value)
    plot_exponential_fit(results[0], save_plot=True)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Omega':>8} {'ν_measured':>12} {'ν_theoretical':>14} {'Error %':>10} {'R²':>8}")
    print("-" * 60)
    
    for result in results:
        omega = result['omega']
        nu_meas = result['nu_measured']
        nu_theo = result['nu_theoretical']
        error = abs(nu_meas - nu_theo) / nu_theo * 100
        r_squared = result['fit_quality']
        
        print(f"{omega:8.1f} {nu_meas:12.6f} {nu_theo:14.6f} {error:10.2f} {r_squared:8.4f}")
    
    print("="*60)
    print("Notes:")
    print("- ν_measured: viscosity from exponential decay fit")
    print("- ν_theoretical: (1/3) * (1/ω - 0.5) for BGK model")
    print("- Error %: relative error between measured and theoretical")
    print("- R²: coefficient of determination for exponential fit")

if __name__ == "__main__":
    main()