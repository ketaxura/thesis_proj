# crowdnav/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TB3:
    wheel_radius: float = 0.033
    wheel_base: float = 0.160
    max_wheel_omega: float = 6.30   # rad/s
    motor_force: float = 1.8        # N·m per wheel

    @property
    def v_max_cap(self) -> float:
        # hardware cap ≈ r * ω_max
        return self.wheel_radius * self.max_wheel_omega

@dataclass
class ControlCfg:
    kp_align: float = 1.2
    omega_align_max: float = 0.8
    align_eps: float = 0.15
    align_hold: int = 10
    omega_max: float = 2.4
    v_go_min: float = 0.05
    v_max_soft: float = 0.20

@dataclass
class LidarCfg:
    num_rays: int = 32
    fov_deg: float = 270
    max_range: float = 8.0
    height: float = 0.175

@dataclass
class Waypointing:
    lookahead_wp: int = 2
    replan_every: int = 10

@dataclass
class WatchdogCfg:
    progress_eps: float = 1e-3
    stall_window: int = 15
    vmin_base: float = 0.03
    vmin_boost: float = 0.08
    vmin_boost_horizon: int = 30

@dataclass
class MPCWeights:
    w_track: float = 5.0
    w_goal: float = 5.0
    w_smooth: float = 0.10
    w_obs: float = 0.2
    w_theta: float = 0.6
    speed_w_base: float = 0.1
    speed_w_boost: float = 0.3
