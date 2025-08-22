# crowdnav/control/track.py
import numpy as np
from ..kinematics import v_omega_to_wheels, clamp_wheels

class TrackController:
    def __init__(self, mpc_runner, tb3_cfg):
        self.mpc = mpc_runner
        self.tb3 = tb3_cfg

    def step(self,
             state,
             path_xy,
             theta_ref,
             obs_traj,
             static_flat,
             weights,
             v_min_soft,
             v_max,
             omega_max,
             dyn,
             T):
        """
        Run one MPC solve, convert (v, ω) to wheel omegas, apply for one control interval.
        Returns: v, omega, U, mpc_dbg
        """

        # ---- Solve MPC (this was mistakenly self.track.step) ----
        v, omega, U, mpc_dbg = self.mpc.solve(
            state,
            path_xy,
            theta_ref,
            obs_traj,    # keep arg present even if unused inside solver for now
            static_flat, # idem
            weights,
            v_min_soft,
            v_max,
            omega_max
        )

        # ---- Wheel rates -> send to motors ----
        w_left, w_right = v_omega_to_wheels(v, omega, self.tb3)
        w_left, w_right = clamp_wheels(w_left, w_right, self.tb3)
        dyn.set_wheel_omegas(w_left, w_right)

        # ---- Advance physics exactly one control interval ----
        dyn.step(T)

        return v, omega, U, mpc_dbg
