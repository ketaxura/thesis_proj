# crowdnav/planning/path_mgr.py
import numpy as np
from ..utils2.coord import grid_to_world

class PathManager:
    def __init__(self, grid, resolution, lookahead_wp=2, replan_every=10, lookahead_m=0.6):
        self.grid = grid
        self.res = resolution
        self.lookahead_wp = lookahead_wp
        self.lookahead_m = float(lookahead_m)
        self.replan_every = replan_every  # 0/None => no periodic replans
        self.global_idx = []
        self._world_cache = None

    def world_path(self):
        if self._world_cache is None and self.global_idx:
            self._world_cache = np.array(
                [grid_to_world(idx, self.res, self.grid.shape) for idx in self.global_idx],
                dtype=float,
            )
        return self._world_cache if self._world_cache is not None else np.empty((0,2))

    def maybe_replan(self, step_count, robot_xy, goal_xy, force=False):
            """Replan on schedule; if replan_every is 0/None, only replan at step 0 or when force=True."""
            periodic = (self.replan_every is not None and self.replan_every > 0)

            if periodic:
                if (step_count % self.replan_every) != 0 and self.global_idx and not force:
                    return False
            else:
                # plan once at reset; skip later unless forced
                if step_count != 0 and self.global_idx and not force:
                    return False

            from ..utils import update_astar_path
            self.global_idx = update_astar_path(robot_xy, goal_xy, self.grid, self.res) or []
            self._world_cache = None
            return bool(self.global_idx)
        
        
        
    # ---- helpers for geometric progress ----
    def _closest_point_on_polyline(self, xy, W):
        """Return (s_best, i_seg, proj_xy) where s_best is arc-length along W."""
        xy = np.asarray(xy, float)
        s_accum = 0.0
        best = (1e9, 0, W[0].copy(), 0.0)  # (dist, seg_idx, proj, s_here)
        for i in range(len(W)-1):
            a, b = W[i], W[i+1]
            ab = b - a
            L2 = float(np.dot(ab, ab)) + 1e-12
            t  = float(np.clip(np.dot(xy - a, ab) / L2, 0.0, 1.0))
            proj = a + t*ab
            d = float(np.linalg.norm(xy - proj))
            s_here = s_accum + t * float(np.linalg.norm(ab))
            if d < best[0]:
                best = (d, i, proj, s_here)
            s_accum += float(np.linalg.norm(ab))
        return best[3], best[1], best[2]  # (s_best, seg_idx, proj_xy)

    def _advance_by_arc(self, W, i0, s0, ds):
        """Advance forward by arc-length ds from (segment i0, cur arc s0). Return target index j."""
        # walk forward until cumulative >= ds
        j = i0
        remain = ds
        # subtract remaining distance in current segment (from projection to end of segment)
        seg_len = float(np.linalg.norm(W[i0+1] - W[i0])) if i0 < len(W)-1 else 0.0
        off_in_seg = min(seg_len, max(0.0, seg_len - (seg_len - (s0 - np.sum(np.linalg.norm(np.diff(W[:i0+1], axis=0), axis=1)) if i0>0 else s0))))
        # (the above “off_in_seg” is conservative; being exact isn’t critical)
        while j < len(W)-1 and remain > 0.0:
            step = float(np.linalg.norm(W[j+1] - W[j]))
            if step >= remain:
                # stop within this segment
                return j+1
            remain -= step
            j += 1
        return min(j+1, len(W)-1)

    def sample_window(self, robot_xy, N):
        W = self.world_path()
        if len(W) == 0:
            return np.repeat(np.asarray(robot_xy, float)[None,:], N, axis=0), np.zeros(N)

        # 1) find along-track s and segment index by projection
        s_here, i_seg, _ = self._closest_point_on_polyline(robot_xy, W)

        # 2) choose a target index by advancing lookahead_m meters ahead along the polyline
        la = float(self.lookahead_m)
        j = self._advance_by_arc(W, i_seg, s_here, la)

        # 3) build the N-point window starting at j
        end = min(j + N, len(W))
        sel = list(range(j, end))
        if len(sel) < N:
            sel += [len(W)-1] * (N - len(sel))
        path_xy = W[sel]

        # 4) headings
        dxy = np.diff(path_xy, axis=0)
        if dxy.shape[0] == 0:
            theta_ref = np.zeros(N)
        else:
            theta_ref = np.concatenate([np.arctan2(dxy[:,1], dxy[:,0]),
                                        [np.arctan2(dxy[-1,1], dxy[-1,0])]])
        return path_xy, theta_ref

    # (optional) progress for HUD
    def progress(self, robot_xy):
        W = self.world_path()
        if len(W) == 0: return 0, 0
        _, i_seg, _ = self._closest_point_on_polyline(robot_xy, W)
        return i_seg, len(W)-1
