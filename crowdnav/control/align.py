# crowdnav/control/align.py
import numpy as np
from ..config import ControlCfg, TB3
from ..kinematics import v_omega_to_wheels, clamp_wheels

class AlignController:
    def __init__(self, cfg: ControlCfg, tb3: TB3):
        self.cfg, self.tb3 = cfg, tb3
        self._hold = 0

    def step(self, theta_err: float, dynamics, T: float) -> bool:
        omega = float(np.clip(self.cfg.kp_align*theta_err, -self.cfg.omega_align_max, self.cfg.omega_align_max))
        wL, wR = v_omega_to_wheels(0.0, omega, self.tb3)
        wL, wR = clamp_wheels(wL, wR, self.tb3)
        dynamics.set_wheel_omegas(wL, wR)
        dynamics.step(T)
        self._hold = self._hold + 1 if abs(theta_err) < self.cfg.align_eps else 0
        return self._hold >= self.cfg.align_hold
