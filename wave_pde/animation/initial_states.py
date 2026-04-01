from dataclasses import dataclass

import numpy as np

from pulse import get_gaussian_pulse


@dataclass(frozen=True)
class WaveStateBase:
    """Base configuration for wave simulations."""
    MAX_AMPLITUDE: float = 1.0
    STEPS_PER_FRAME: int = 25
    N: int = 100
    dx: float = 1.0
    dt: float = 0.01
    c: float = 10.0

    @property
    def C(self):
        """Courant number calculation."""
        return self.c * self.dt / self.dx

@dataclass(frozen=True)
class InitialStates1D(WaveStateBase):
    N: int = 100
    c: float = 8

    @property
    def is_stable(self) -> bool:
        """
        CFL Condition for 1D: C <= 1.0
        """
        return self.C <= 1.0

    @property
    def initial_u(self):
        """Generate initial wave state with multiple Gaussian pulses."""
        u = np.zeros(self.N)
        # u += get_gaussian_pulse(self.N, self.MAX_AMPLITUDE / 2, 2.5, center=self.N // 4)  
        # u += get_gaussian_pulse(self.N, self.MAX_AMPLITUDE / 2, 2.5, center=3 * self.N // 4)
        u += get_gaussian_pulse(self.N, self.MAX_AMPLITUDE)
        return u


    pass

@dataclass(frozen=True)
class InitialStates2D(WaveStateBase):
    N: int = 500
    c: float = 8.0
    WINDOW_SIZE: int = 500
    STEPS_PER_FRAME: int = 20
    MAX_AMPLITUDE: float = 1

    @property
    def is_stable(self) -> bool:
        """CFL condition for 2D: C <= 1/sqrt(2)"""
        return self.C <= 1 / np.sqrt(2)

    @property
    def initial_u(self):
        """Generate initial wave state with multiple Gaussian pulses."""
        dim = (self.N, self.N)
        u = np.zeros(dim)
        # u += get_gaussian_pulse(dim, self.MAX_AMPLITUDE / 2, 10, center=[self.N // 4, self.N // 4])
        # u += get_gaussian_pulse(dim, self.MAX_AMPLITUDE / 2, 10, center=[3 * self.N // 4, 3 * self.N // 4])
        # u += get_gaussian_pulse(dim, self.MAX_AMPLITUDE)
        # u += get_gaussian_pulse(dim, self.MAX_AMPLITUDE / 2, 10, center=[self.N // 4, 3 * self.N // 4])
        # u += get_gaussian_pulse(dim, self.MAX_AMPLITUDE / 2, 10, center=[3 * self.N // 4, self.N // 4])
        return u