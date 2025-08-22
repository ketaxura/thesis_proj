# crowdnav/utils2/watchdog.py
import numpy as np

class ProgressWatchdog:
    def __init__(self, progress_eps=1e-3, stall_window=15, vmin_base=0.03, vmin_boost=0.08, vmin_boost_horizon=30):
        self.progress_eps = progress_eps
        self.stall_window = stall_window
        self.vmin_base = vmin_base
        self.vmin_boost = vmin_boost
        self.vmin_boost_horizon = vmin_boost_horizon
        self.stall_ctr = 0
        self._dg_prev = None
        self._boost_ctr = 0

    def update(self, d_goal: float):
        if self._dg_prev is not None and (self._dg_prev - d_goal) < self.progress_eps:
            self.stall_ctr += 1
        else:
            self.stall_ctr = 0
        self._dg_prev = d_goal
        if self.stall_ctr >= self.stall_window and self._boost_ctr == 0 and d_goal > 0.3:
            self._boost_ctr = self.vmin_boost_horizon

    def vmin_soft(self):
        if self._boost_ctr > 0:
            self._boost_ctr -= 1
            return self.vmin_boost, True
        return self.vmin_base, False
