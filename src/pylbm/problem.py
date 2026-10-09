"""Problem description shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lattice import D2Q9, D3Q19, Lattice, magic_rate, mrt_rates

COLLISIONS = ("bgk", "trt", "mrt")


@dataclass
class Problem:
    """A lattice-Boltzmann problem on a rectangular box.

    ``shape`` is (nx, ny) or (nx, ny, nz). Each axis is either periodic or bounded by
    half-way bounce-back walls. The top wall (high side of the last axis) moves with
    velocity ``u_lid`` in +x (Ladd momentum correction, wall density rho_w = 1).

    The general (NumPy / PyTorch) solvers additionally support solid obstacles, a velocity
    inlet on the left wall and a zero-gradient outlet on the right; fused GPU kernels
    support the plain box only (see :meth:`is_box`).
    """

    shape: tuple[int, ...]
    tau: float = 0.6
    collision: str = "bgk"
    periodic: tuple[bool, ...] | None = None
    u_lid: float = 0.0
    force: tuple[float, ...] | None = None
    magic: float = 3.0 / 16.0  # TRT magic parameter Lambda
    s_e: float = 1.64  # MRT bulk-type rates (free parameters)
    s_eps: float = 1.54
    # --- general-solver extras -------------------------------------------------------
    solid: np.ndarray | None = None  # bool, array layout (ny, nx)
    inlet: np.ndarray | None = None  # u_x(y) on the left wall, shape (ny,)
    outlet: bool = False  # zero-gradient outflow on the right boundary
    init_rho: np.ndarray | None = None  # array layout
    init_u: np.ndarray | None = None  # (d, *array layout)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        if self.dim not in (2, 3):
            raise ValueError("shape must be 2D or 3D")
        if self.periodic is None:
            self.periodic = (False,) * self.dim
        self.periodic = tuple(bool(p) for p in self.periodic)
        if self.force is None:
            self.force = (0.0,) * self.dim
        self.force = tuple(float(v) for v in self.force)
        if self.collision not in COLLISIONS:
            raise ValueError(f"collision must be one of {COLLISIONS}")
        if self.collision == "mrt" and self.dim != 2:
            raise ValueError("MRT is implemented for D2Q9 only")
        if self.u_lid and self.periodic[-1]:
            raise ValueError("a moving lid needs a wall on the last axis")
        if not 0.5 < self.tau:
            raise ValueError("tau must be > 0.5")

    # ------------------------------------------------------------------ properties
    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def lattice(self) -> Lattice:
        return D2Q9 if self.dim == 2 else D3Q19

    @property
    def array_shape(self) -> tuple[int, ...]:
        """Spatial array shape (slowest axis first): (ny, nx) or (nz, ny, nx)."""
        return tuple(reversed(self.shape))

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nu(self) -> float:
        """Kinematic viscosity in lattice units."""
        return (self.tau - 0.5) / 3.0

    @property
    def has_force(self) -> bool:
        return any(v != 0.0 for v in self.force)

    def is_box(self) -> bool:
        """True if the problem only uses features the fused GPU kernels implement."""
        return self.solid is None and self.inlet is None and not self.outlet

    def trt_rates(self) -> tuple[float, float]:
        """(s_plus, s_minus) for TRT."""
        s_plus = 1.0 / self.tau
        return s_plus, magic_rate(s_plus, self.magic)

    def mrt_rates(self) -> np.ndarray:
        return mrt_rates(self.tau, self.s_e, self.s_eps)

    def mrt_rate_tuple(self) -> tuple[float, ...]:
        """MRT rates as plain floats (safe inside torch.compile-traced code)."""
        s_nu = 1.0 / self.tau
        s_q = magic_rate(s_nu, 3.0 / 16.0)
        return (0.0, self.s_e, self.s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu)

    def initial_populations(self) -> np.ndarray:
        """Equilibrium populations for the initial fields (float64, SoA layout)."""
        from .lattice import equilibrium

        shp = self.array_shape
        rho = np.ones(shp) if self.init_rho is None else np.asarray(self.init_rho, float)
        u = np.zeros((self.dim,) + shp) if self.init_u is None else np.asarray(self.init_u, float)
        return equilibrium(self.lattice, rho, u)

    @staticmethod
    def cavity(n: int | tuple[int, ...], re: float | None = None, u_lid: float = 0.1,
               tau: float | None = None, collision: str = "bgk", dim: int = 2) -> Problem:
        """Lid-driven cavity. Give either ``re`` (Re = u_lid * n / nu) or ``tau``."""
        shape = (n,) * dim if isinstance(n, int) else tuple(n)
        if tau is None:
            if re is None:
                raise ValueError("give re or tau")
            tau = 3.0 * u_lid * shape[0] / re + 0.5
        return Problem(shape, tau=tau, collision=collision, u_lid=u_lid)
