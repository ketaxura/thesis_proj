# crowdnav/kinematics.py
import numpy as np
from .config import TB3

def v_omega_to_wheels(v: float, omega: float, tb3: TB3):
    r, L = tb3.wheel_radius, tb3.wheel_base
    return float((v - 0.5*omega*L)/r), float((v + 0.5*omega*L)/r)

def clamp_wheels(wL, wR, tb3: TB3):
    lim = tb3.max_wheel_omega
    return float(np.clip(wL, -lim, lim)), float(np.clip(wR, -lim, lim))
