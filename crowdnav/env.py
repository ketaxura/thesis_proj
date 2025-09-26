# crowdnav/env.py (import block)
import gym, numpy as np, pybullet as p, pybullet_data, traceback
import time
from gym import spaces


from .logger_setup import logger
from .config import TB3, ControlCfg, LidarCfg, Waypointing, WatchdogCfg, MPCWeights
from .world import USE_OBJ_MAP, SPAWN_BBOX_START, SPAWN_BBOX_GOAL
from .dynamics import Dynamics
from .sensors import LidarSensor
from .control.align import AlignController
from .control.track import TrackController
from .planning.mpc_runner import MPCRunner
from .planning.path_mgr import PathManager
from .utils2.coord import world_to_grid, grid_to_world, wrap_angle
from .utils2.hud import DebugHUD

# ⬇︎ change these to single-dot since files are in the same package
from .mpc import build_mpc_solver_random_obs
from .world import create_world
from .utils import update_astar_path, MAP_SCALE, map_size




v_max_cap = 0.20
ail=0
REPLAN_PERIOD = 25   # steps ~ 2.5 s if T=0.1





# === Heading helpers (place near other top-level helpers) ===
def _unwrap(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi



def _wrap_pi(a: float) -> float:
    # map to (-π, π]
    return (a + np.pi) % (2*np.pi) - np.pi

def _unwrap_to(prev: float, meas_wrapped: float) -> float:
    # bring 'meas_wrapped' into the local neighborhood of 'prev'
    d = _wrap_pi(meas_wrapped - prev)
    return prev + d


def _smooth_to(prev_vec, target_vec, max_step=0.35):
    """Move from prev_vec to target_vec, limiting per-index change to ±max_step rad."""
    prev = np.asarray(prev_vec, dtype=float)
    tgt  = np.asarray(target_vec, dtype=float)
    d    = tgt - prev
    # wrap each element into (-π,π]
    d = (d + np.pi) % (2*np.pi) - np.pi
    d = np.clip(d, -max_step, +max_step)
    return prev + d

def _ensure_len_N(arr, N):
    """
    Ensure the first dimension has length N.
    - If shorter, repeat the last element.
    - If longer, crop.
    Works for 1D (theta_ref) and 2D (path_xy) arrays.
    """
    a = np.asarray(arr)
    if a.shape[0] < N:
        pad = np.repeat(a[-1:,...], N - a.shape[0], axis=0)
        return np.concatenate([a, pad], axis=0)
    return a[:N]


def _make_theta_ref(path_xy: np.ndarray,
                    theta_prev: np.ndarray | None = None,
                    k_strong: int = 3,
                    alpha: float = 0.25) -> np.ndarray:
    """
    Build a heading reference for the MPC that does NOT encourage early pre-turn.
    Steps:
      1) tangent from forward differences
      2) unwrap (no 2π jumps)
      3) light low-pass vs last cycle (optional)
      4) **preview taper**: beyond the first k_strong steps, freeze the reference
         to the heading of step (k_strong-1). This stops the optimizer from
         'seeing' a far bend and turning too early.
    """
    # 1) tangent of the local path window
    dxy = np.diff(path_xy, axis=0, append=path_xy[-1:])  # last step duplicated
    raw = np.arctan2(dxy[:, 1], dxy[:, 0])

    # Ensure the very first angle matches the first real segment (1 -> 0)
    if len(path_xy) >= 2:
        v10 = path_xy[1] - path_xy[0]
        raw[0] = np.arctan2(v10[1], v10[0])
        # 2) unwrap sequentially
        ref = raw.copy()
        for i in range(1, len(ref)):
            ref[i] = ref[i-1] + _wrap_pi(ref[i] - ref[i-1])
            
            
    # after unwrapping to 'ref'
    if theta_prev is not None and len(theta_prev) == len(ref):
        ref = _smooth_to(theta_prev, ref, max_step=0.35)  # ~20° per cycle


    # 3) temporal smoothing wrt previous ref (optional)
    if theta_prev is not None and len(theta_prev) == len(ref):
        out = np.empty_like(ref)
        prev = float(theta_prev[0])
        for i in range(len(ref)):
            r = prev + _wrap_pi(ref[i] - prev)
            prev = prev + alpha * (r - prev)
            out[i] = prev
        ref = out

    # 4) preview taper (freeze beyond first k_strong indices)
    k_strong = max(1, min(int(k_strong), len(ref)))
    freeze_val = float(ref[k_strong - 1])
    for i in range(k_strong, len(ref)):
        ref[i] = freeze_val

    return ref



def _max_curvature(path_xy: np.ndarray, k_window: int = 6, eps: float = 1e-3) -> float:
    pts = np.asarray(path_xy, dtype=float)
    if pts.shape[0] < 3: 
        return 0.0
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    # NEW: drop tiny segments (e.g., < 5 cm)
    keep = seg_len > 0.05
    seg = seg[keep]
    seg_len = seg_len[keep]
    if seg.shape[0] < 2:
        return 0.0

    hdg = np.unwrap(np.arctan2(seg[:,1], seg[:,0]))
    dtheta = np.abs(np.diff(hdg))
    ds = 0.5 * (seg_len[1:] + seg_len[:-1])
    kappa = dtheta / np.maximum(ds, eps)
    w = min(int(k_window), kappa.size)
    return float(np.max(kappa[:w])) if w > 0 else 0.0


def _poly_cumlen(P: np.ndarray):
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    if len(P) < 2:
        s = np.array([0.0], dtype=float)
        return s, 0.0
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s, float(s[-1])

def _project_s(P: np.ndarray, q: np.ndarray):
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    q = np.asarray(q, dtype=float).reshape(2)
    s_cum, L = _poly_cumlen(P)
    if L <= 1e-9 or len(P) < 2:
        return 0.0, 0.0, {"L": L, "d_min": float("inf")}
    best_s, best_d2 = 0.0, float("inf")
    for i in range(len(P) - 1):
        a, b = P[i], P[i+1]
        v = b - a
        vv = float(np.dot(v, v)) + 1e-12
        t = float(np.clip(np.dot(q - a, v) / vv, 0.0, 1.0))
        p = a + t * v
        d2 = float(np.dot(q - p, q - p))
        s_here = s_cum[i] + t * np.linalg.norm(v)
        if d2 < best_d2:
            best_d2, best_s = d2, s_here
    s_frac = 0.0 if L < 1e-9 else np.clip(best_s / L, 0.0, 1.0)
    return best_s, float(s_frac), {"L": float(L), "d_min": float(np.sqrt(best_d2))}



# ---- Polyline utilities (already in your file; keep) ----
def _poly_cumlen(P):
    import numpy as np
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    if len(P) < 2:
        s = np.array([0.0], dtype=float)
        return s, 0.0
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s, float(s[-1])

def _project_s(P, q):
    import numpy as np
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    q = np.asarray(q, dtype=float).reshape(2)
    s_cum, L = _poly_cumlen(P)
    if L <= 1e-9 or len(P) < 2:
        return 0.0, 0.0, {"L": L, "d_min": float("inf")}
    best_s, best_d2 = 0.0, float("inf")
    for i in range(len(P) - 1):
        a, b = P[i], P[i+1]
        v = b - a
        vv = float(np.dot(v, v)) + 1e-12
        t = float(np.clip(np.dot(q - a, v) / vv, 0.0, 1.0))
        p = a + t * v
        d2 = float(np.dot(q - p, q - p))
        s_here = s_cum[i] + t * np.linalg.norm(v)
        if d2 < best_d2:
            best_d2, best_s = d2, s_here
    s_frac = 0.0 if L < 1e-9 else np.clip(best_s / L, 0.0, 1.0)
    return best_s, float(s_frac), {"L": float(L), "d_min": float(np.sqrt(best_d2))}

def _eval_poly_at_s(P, s_query):
    """Linear interpolation on a piecewise-linear path by arc-length."""
    import numpy as np
    P = np.asarray(P, dtype=float).reshape(-1, 2)
    s_cum, L = _poly_cumlen(P)
    s = np.clip(float(s_query), 0.0, L)
    # find segment i such that s in [s_i, s_{i+1}]
    i = int(np.searchsorted(s_cum, s, side='right') - 1)
    i = max(0, min(i, len(P) - 2))
    ds = s - s_cum[i]
    seg = P[i+1] - P[i]
    seg_len = float(np.linalg.norm(seg)) + 1e-12
    t = ds / seg_len
    return P[i] + t * seg





class CrowdNavPyBulletEnv(gym.Env):
    def __init__(self, num_peds=0, max_static=8, max_steps=2000, resolution=0.1, seed: int | None = None):
        super().__init__()
        logger.info("CrowdNavPyBulletEnv initializing...")
        self.client = p.connect(p.GUI)
        
        
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide right-side Params & top bars
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)      # optional
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0) # optional
        
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self._dbg_path_ids = []   # pybullet debug item ids for the A* path

        self.hud = DebugHUD()
        # config objects
        self.tb3   = TB3()
        self.ctrl  = ControlCfg()
        self.lcfg  = LidarCfg()
        self.wpcfg = Waypointing()
        self.wdog  = WatchdogCfg()
        self.weights = MPCWeights()
        self._rng = np.random.default_rng(seed)
        self._last_seed = seed
        
        self._s_star = 0.0      # monotone progress along current path
        self._s_last = 0.0      # for debugging/guard
        self._path_L = 0.0

        
        self._last_replan_step = -10**9

        self.resolution = resolution
        self.world_size = map_size * MAP_SCALE
        self.half_size  = self.world_size / 2
        self.max_steps  = max_steps
        self.num_peds   = int(num_peds)
        self.max_static = int(max_static)
        
        
        self._omega_sat_streak = 0
        self._last_wp_idx = 0
        self._force_pivot = False


        # world & robot
        (self.robot_id, self.left_wheel_joint_id, self.right_wheel_joint_id,
         self.grid, self.static_obs) = create_world(self.client, resolution=self.resolution)

        self.dyn = Dynamics(self.robot_id, self.left_wheel_joint_id, self.right_wheel_joint_id, self.tb3.motor_force)
        self.lidar = LidarSensor(self.lcfg)

        # MPC
        solver, f_dyn, self.T, self.N = build_mpc_solver_random_obs(max_obs=self.num_peds, max_static=self.max_static)
        self.mpc  = MPCRunner(solver, self.T, self.N, f_dyn=f_dyn)  # <— add f_dyn
        self.track= TrackController(self.mpc, self.tb3)
        self.align= AlignController(self.ctrl, self.tb3)
        self.path = PathManager(self.grid, self.resolution, replan_every=0, lookahead_m=0.4)



        # observation/action spaces
        obs_dim = self.lcfg.num_rays + 2 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.phase = "ALIGN"
        self._align_ctr = 0
        self.step_count = 0
        self.robot_pos = np.array([1.0, 1.0])
        self.theta = 0.0          # keep if other code uses it
        self.theta_cont = 0.0     # <-- new: continuous heading
        self.goal_pos = np.array([4.0, 4.0])
        self._theta_ref_prev = None
        
        
        self._spawn_rect = {
            "start": SPAWN_BBOX_START if USE_OBJ_MAP else None,
            "goal":  SPAWN_BBOX_GOAL  if USE_OBJ_MAP else None,
        }


        # limits
        self.v_max = min(v_max_cap, self.tb3.v_max_cap)

        logger.info(f"World ready | grid: {self.grid.shape}, res: {self.resolution}, N={self.N}, T={self.T}")

    # grid/world helpers
    def world_to_grid(self, pos): return world_to_grid(pos, self.resolution, self.grid.shape)
    def grid_to_world(self, idx): return grid_to_world(idx, self.resolution, self.grid.shape)
    
    
    
    
    
    def _draw_astar_path(self, world_pts, vertical_count=10, connect=True,
                        label_every=1, z=0.05, text_size=1.2,
                        line_rgb=(0,0,1), text_rgb=(1,1,0)):
        """Draw vertical blue lines + connecting polyline + numeric labels per waypoint.

        Args:
            world_pts: iterable of (x,y) in world coords
            vertical_count: draw vertical pillars for the first N waypoints
            connect: draw a ground polyline connecting all waypoints
            label_every: label every k-th waypoint (1 = label all)
            z: text height above ground
            text_size: pybullet text size
            line_rgb: color for path lines
            text_rgb: color for waypoint numbers
        """
        
        
        # clear old items
        for i in self._dbg_path_ids:
            try: p.removeUserDebugItem(i)
            except Exception: pass
        self._dbg_path_ids.clear()

        # robust empty check (handles list/np.array)
        if world_pts is None:
            return
        try:
            n_pts = len(world_pts)
        except TypeError:
            # not sized, try to iterate once
            world_pts = list(world_pts)
            n_pts = len(world_pts)
        if n_pts == 0:
            return

        # ensure array-like with shape (M,2)
        pts = np.asarray(world_pts, dtype=float).reshape(-1, 2)


        
        # vertical markers
        for w in pts[:max(0, 1)]:
            x, y = float(w[0]), float(w[1])
            uid = p.addUserDebugLine([x, y, 0], [x, y, 10], line_rgb, lineWidth=2, lifeTime=0)
            self._dbg_path_ids.append(uid)

        # optional connecting polyline near ground
        if connect and len(pts) >= 2:
            for a, b in zip(pts[:-1], pts[1:]):
                uid = p.addUserDebugLine([float(a[0]), float(a[1]), 0.01],
                                        [float(b[0]), float(b[1]), 0.01],
                                        line_rgb, lineWidth=2, lifeTime=0)
                self._dbg_path_ids.append(uid)
                
        

        # numeric labels
        if label_every is None or label_every <= 0:
            label_every = 1
        
        
        
        # for i, (x, y) in enumerate(pts):
        #     if (i % label_every) != 0:
        #         continue
        #     uid = p.addUserDebugText(
        #         text=str(i),
        #         textPosition=[float(x), float(y), float(z)],
        #         textColorRGB=text_rgb,
        #         textSize=float(text_size),
        #         lifeTime=0
        #     )
        #     self._dbg_path_ids.append(uid)
            
            
    def _draw_spawn_rect(self, rect, rgb):
        if rect is None: return
        x0, y0, x1, y1 = rect
        pts = [(x0,y0,0.02),(x1,y0,0.02),(x1,y1,0.02),(x0,y1,0.02)]
        for a,b in zip(pts, pts[1:]+pts[:1]):
            p.addUserDebugLine(a,b,rgb, lineWidth=2, lifeTime=0)


            
        
    def find_free_grid(self, label, avoid=None, min_dist=0.5, max_attempts=200, border_margin=0.2):
        """
        Sample a free world cell for <label> within an optional spawn rectangle.
        If no rectangle, sample the whole grid extent (your current behavior).
        """
        rect = self._spawn_rect.get(label, None)

        def _sample_world_xy():
            if rect is not None:
                x0, y0, x1, y1 = rect
                # normalize (in case inputs are swapped)
                lo_x, hi_x = (min(x0, x1), max(x0, x1))
                lo_y, hi_y = (min(y0, y1), max(y0, y1))
                x = self._rng.uniform(lo_x, hi_x)
                y = self._rng.uniform(lo_y, hi_y)
                return x, y
            else:
                # fallback: derive from grid shape (centered-at-origin assumption)
                H, W = self.grid.shape
                half_x = 0.5 * W * self.resolution
                half_y = 0.5 * H * self.resolution
                m = float(border_margin)
                x = self._rng.uniform(-half_x + m, +half_x - m)
                y = self._rng.uniform(-half_y + m, +half_y - m)
                return x, y

        for _ in range(int(max_attempts)):
            x, y = _sample_world_xy()
            r, c = self.world_to_grid((x, y))
            # bounds & free check
            if r < 0 or r >= self.grid.shape[0] or c < 0 or c >= self.grid.shape[1]:
                continue
            if self.grid[r, c] != 0:
                continue
            w = self.grid_to_world((r, c))
            if avoid is not None:
                ref = avoid if isinstance(avoid, np.ndarray) else np.array(avoid, dtype=float)
                if np.linalg.norm(w - ref) < float(min_dist):
                    continue
            return (r, c)

        raise ValueError(f"Could not find free {label} cell after {max_attempts} tries")




    def reset(self, visualize: bool = False,
              randomize: bool = True,
              min_start_goal_dist: float = 2.0,
              max_place_attempts: int = 40):
        """
        If randomize=True, pick random free start/goal with a valid A* path and
        a minimum separation in meters.
        
        
        """
        self._theta_ref_prev = None
        self.step_count = 0
        self.phase = "ALIGN"; self._align_ctr = 0
        self._last_replan_step = -10**9
        
        
        self._omega_sat_streak = 0
        self._last_wp_idx = 0
        self._force_pivot = False
        
        
            # call in reset(), once per episode
        if USE_OBJ_MAP:
            self._draw_spawn_rect(self._spawn_rect["start"], (0,0,1))
            self._draw_spawn_rect(self._spawn_rect["goal"],  (0,1,0))

        # time.sleep(2000) #REMOVE THIS WHEN FOR REALSIES


        # (optional) clear old debug path items
        for i in getattr(self, "_dbg_path_ids", []):
            try: p.removeUserDebugItem(i)
            except: pass
        self._dbg_path_ids = []


        # from .utils2.coord import _coord_self_test
        # _coord_self_test(self.grid.shape, self.resolution)
        # Choose start/goal
        if randomize:
            ok = False
            for _ in range(max_place_attempts):
                # sample start
                start_idx = self.find_free_grid("start")
                start_w   = self.grid_to_world(start_idx)

                # sample goal (far enough)
                goal_idx  = self.find_free_grid("goal", avoid=start_w, min_dist=min_start_goal_dist)
                goal_w    = self.grid_to_world(goal_idx)

                # try A* once
                self.start_idx, self.goal_idx = start_idx, goal_idx
                self.robot_pos = start_w
                self.goal_pos  = goal_w

                # give a random initial heading
                self.theta = float(self._rng.uniform(-np.pi, np.pi))
                self.theta_cont = self.theta

                # place robot + goal pole
                self.dyn.reset_pose(self.robot_pos, self.theta, z=0.008)
                p.addUserDebugLine([*self.goal_pos,0],[*self.goal_pos,10],[0,1,0], lineWidth=3, lifeTime=0)

                # A*: if path exists, keep; else retry
                replanned = self.path.maybe_replan(0, self.robot_pos, self.goal_pos)
                if replanned:
                    ok = True
                    break

            if not ok:
                raise RuntimeError("Failed to sample a reachable start/goal after retries")
        else:
            # your previous fixed start/goal
            manual_start = (18, 88); manual_goal = (14, 22)
            if self.grid[manual_start] == 1 or self.grid[manual_goal] == 1:
                raise ValueError("Start/Goal inside obstacle!")
            self.start_idx = manual_start; self.goal_idx = manual_goal
            self.robot_pos = self.grid_to_world(self.start_idx)
            self.goal_pos  = self.grid_to_world(self.goal_idx)
            self.theta = 0.0
            self.theta_cont = 0.0
            self.dyn.reset_pose(self.robot_pos, self.theta, z=0.008)
            p.addUserDebugLine([*self.goal_pos,0],[*self.goal_pos,10],[0,1,0], lineWidth=3, lifeTime=0)
            replanned = self.path.maybe_replan(0, self.robot_pos, self.goal_pos)
            
            
        

        # after: replanned = self.path.maybe_replan(0, self.robot_pos, self.goal_pos)
        # and before drawing the path
        path_full = self.path.world_path()
        if path_full is not None and len(path_full) >= 2:
            s0, _, info = _project_s(np.asarray(path_full), self.robot_pos)
            self._path_L = float(info["L"])
            self._s_star = max(0.0, min(self._path_L, s0 - 0.05))  # tiny back hysteresis
        else:
            self._s_star = 0.0
            self._path_L = 0.0


        # Draw the path if available
        if replanned:
            
            self._draw_astar_path(self.path.world_path(),
                                  vertical_count=10, connect=True,
                                  label_every=1, text_size=1.2)


        # wheel dynamics tuning
        try:
            self.dyn.tune_wheels(lateral_friction=0.6, rolling_friction=0.0,
                                 spinning_friction=0.0, linear_damping=0.02, angular_damping=0.02)
        except Exception:
            pass

        return self._get_observation()

    def _update_pose(self):
        pos, theta_wrapped = self.dyn.get_pose()  # what units does this return?
        # ---- Unit guard: convert deg → rad if someone fed degrees ----
        if abs(theta_wrapped) > 2*np.pi*1.5:      # bigger than ~9.4 rad ⇒ not radians
            theta_wrapped = np.deg2rad(theta_wrapped)
        # --------------------------------------------------------------
        self.theta_cont = _unwrap_to(self.theta_cont, float(theta_wrapped))
        self.robot_pos = np.array(pos, dtype=float)
        return self.robot_pos, self.theta_cont


    def _get_observation(self):
        lidar = self.lidar.scan(self.robot_id)
        pos, _, _ = self.dyn.get_pose_full()
        goal_vec  = self.goal_pos - pos[:2]
        goal_dist = np.linalg.norm(goal_vec) + 1e-6
        goal_dir  = goal_vec / goal_dist
        return np.concatenate([lidar, goal_dir, [goal_dist]]).astype(np.float32)

    def step(self, action):
        try:
            # time.sleep(2000) #REMOVE THIS WHEN FOR REALSIES
            
            
            self.step_count += 1
            print(self.step_count)
            self._update_pose()
            state = np.array([*self.robot_pos, self.theta_cont], dtype=float)



            # --------- only TRACK below this line ---------

            # Replan cadence + hysteresis
            if not hasattr(self, "_last_plan_pose"):
                self._last_plan_pose = self.robot_pos.copy()
                self._last_plan_step = -999

            REPLAN_EVERY_STEPS = int(round(5 / self.T))  # e.g., every 0.5 s
            REPLAN_MOVE_THRESH = 0.08                      # replan if moved ≥ 8 cm
            time_ok = (self.step_count - self._last_plan_step) >= REPLAN_EVERY_STEPS
            move_ok = np.linalg.norm(self.robot_pos - self._last_plan_pose) >= REPLAN_MOVE_THRESH

            # ----- build forward ribbon from monotone arc-length (exactly N points) -----
            path_full = self.path.world_path()
            if (path_full is None) or (len(path_full) < 2):
                # fallback: straight line to goal with N samples
                path_xy = np.linspace(self.robot_pos, self.goal_pos, self.N).astype(float)
            else:
                s_now, _, info = _project_s(np.asarray(path_full), self.robot_pos)
                self._path_L = float(info["L"])

                # monotone progress with tiny back hysteresis
                BACK_HYST = 0.03
                self._s_star = max(self._s_star, s_now - BACK_HYST)
                self._s_star = float(np.clip(self._s_star, 0.0, self._path_L))

                LOOKAHEAD = 0.40
                s_head = float(min(self._path_L, self._s_star + 0.08))  # carrot offset
                s_tail = float(min(self._path_L, s_head + LOOKAHEAD))
                if s_tail <= s_head + 1e-9:
                    s_grid = np.full(self.N, s_head, dtype=float)        # degenerate near goal
                else:
                    s_grid = np.linspace(s_head, s_tail, self.N, dtype=float)

                path_xy = np.array([_eval_poly_at_s(path_full, s) for s in s_grid], dtype=float)


            theta_ref  = _make_theta_ref(path_xy, self._theta_ref_prev, k_strong=3, alpha=0.25)
            self._theta_ref_prev = theta_ref.copy()

            # ALIGN uses tangent (not point-to-waypoint)
            theta_des0  = float(theta_ref[0])
            theta_error = _wrap_pi(theta_des0 - self.theta_cont)
            if self.phase == "ALIGN":
                if self.align.step(theta_error, self.dyn, self.T):
                    self.phase = "TRACK"
                self._update_pose()
                dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
                done = (dg < 0.2) or (self.step_count >= self.max_steps)
                reward = 1.0 if dg < 0.2 else -0.01
                return self._get_observation(), reward, done, {}


            # progress watchdog (unchanged)
            dgoal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            from .utils2.watchdog import ProgressWatchdog
            if not hasattr(self, "_wd"):
                self._wd = ProgressWatchdog(self.wdog.progress_eps, self.wdog.stall_window,
                                            self.wdog.vmin_base, self.wdog.vmin_boost, self.wdog.vmin_boost_horizon)
            self._wd.update(dgoal)
            v_min_soft, boosted = self._wd.vmin_soft()

            # ----- build MPC params (unchanged) -----
            obs_vec = self._get_observation()
            lidar = obs_vec[:self.lcfg.num_rays]
            pos, quat, R = self.dyn.get_pose_full()
            origins = np.repeat(pos[None,:], self.lcfg.num_rays, axis=0)
            dirs_world = (R @ self.lidar.dirs.T).T
            hit_pts = origins + (lidar[:,None]*self.lcfg.max_range)*dirs_world
            valid = lidar < 1.0
            pts2d = hit_pts[valid,:2]
            order = np.argsort(np.linalg.norm(pts2d - self.robot_pos, axis=1))
            closest = pts2d[order][:self.max_static]
            static_flat = closest.flatten()
            if static_flat.size < 2*self.max_static:
                static_flat = np.concatenate([static_flat, np.full(2*self.max_static - static_flat.size, 1e6)])

            if self.num_peds > 0:
                obs_traj = np.zeros(2*self.N*self.num_peds)
            else:
                obs_traj = np.zeros(2*self.N*self.num_peds)

            weights = np.array([
                self.weights.w_track,
                self.weights.w_goal,
                self.weights.w_smooth,
                self.weights.w_obs,
                (self.weights.speed_w_boost if boosted else self.weights.speed_w_base),
                self.weights.w_theta
            ], dtype=float)



            dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            R_SLOW, R_STOP = 0.80, 0.1

            # curvature cap
            kappa_max = _max_curvature(path_xy, k_window=6, eps=1e-3)
            vcap_curv = self.v_max if kappa_max <= 1e-3 else self.ctrl.omega_max / max(kappa_max, 1e-6)

            # goal funnel
            v_cap_goal = self.v_max if dist_to_goal >= R_SLOW else self.v_max * (dist_to_goal / R_SLOW)

            vmax_local = float(np.clip(min(v_cap_goal, vcap_curv, self.v_max), 0.03, self.v_max))
            if dist_to_goal < 0.1:
                v_min_soft = 0.0

            if dist_to_goal < R_STOP:
                self.dyn.set_cmd(0.0, 0.0)
                return self._get_observation(), 1.0, True, {"reason": "goal_reached"}


            # === TRACK (MPC) ===
            v, omega, U, mpc_dbg = self.track.step(
                state, path_xy, theta_ref, obs_traj, static_flat, weights,
                v_min_soft, vmax_local, self.ctrl.omega_max, self.dyn, self.T
            )

            # pose update and progress
            self._update_pose()
            
            
        
            # --- Arc-length progress & termination (robust) ---
            # ----- stable path window from arc-length (no replan here) -----
            path_full = self.path.world_path()
            if path_full is None or len(path_full) < 2:
                # if we lost the path, do a single replan to seed it
                if self.path.maybe_replan(self.step_count, self.robot_pos, self.goal_pos):
                    path_full = self.path.world_path()

            if path_full is not None and len(path_full) >= 2:
                # update arc-length projection, then enforce monotonic s*
                s_now, _, info = _project_s(np.asarray(path_full), self.robot_pos)
                self._path_L = float(info["L"])
                # never decrease; allow tiny retreat (numerical) with hysteresis
                BACK_HYST = 0.03
                self._s_star = max(self._s_star, s_now - BACK_HYST)
                self._s_star = float(np.clip(self._s_star, 0.0, self._path_L))

                # build preview strictly ahead of s*
                LOOKAHEAD = 0.40   # meters of path to feed MPC
                DS        = max(0.05, LOOKAHEAD / max(2, self.N - 1))
                s_head    = float(self._s_star + 0.08)  # small positive offset removes tangent flips
                s_tail    = float(min(self._path_L, s_head + LOOKAHEAD))
                samples   = int(1 + round((s_tail - s_head) / DS))
                s_grid    = s_head + DS * np.arange(samples, dtype=float)
                path_xy   = np.array([_eval_poly_at_s(path_full, s) for s in s_grid], dtype=float)

                # tangent-based heading ref from this forward ribbon
                theta_ref = _make_theta_ref(path_xy, self._theta_ref_prev, k_strong=3, alpha=0.25)
                self._theta_ref_prev = theta_ref.copy()

                # alignment target = tangent at the first sample (not “pointing to” a waypoint)
                theta_des0  = float(theta_ref[0])
                theta_error = _wrap_pi(theta_des0 - self.theta_cont)
            else:
                # safety fallback: aim at goal
                path_xy   = np.array([self.robot_pos, self.goal_pos], dtype=float)
                theta_ref = _make_theta_ref(path_xy, self._theta_ref_prev, k_strong=1, alpha=0.25)
                theta_des0  = float(theta_ref[0])
                theta_error = _wrap_pi(theta_des0 - self.theta_cont)

            # ---- GUARANTEE exact N for the solver ----
            path_xy   = _ensure_len_N(path_xy,   self.N)
            theta_ref = _ensure_len_N(theta_ref, self.N)
            # Optional quick checks (safe to comment out later)
            # assert path_xy.shape == (self.N, 2),  f"path_xy shape {path_xy.shape}"
            # assert theta_ref.shape == (self.N,),   f"theta_ref shape {theta_ref.shape}"
                        


            # --- termination & reward (now variables are defined) ---
            s_goal = float(self._path_L)
            s_star = float(self._s_star)
            s_frac = 0.0 if s_goal <= 1e-9 else s_star / s_goal
            dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))

            done = (dist_to_goal < R_STOP) or (s_goal > 0 and s_star >= s_goal - 0.05) or (self.step_count >= self.max_steps)
            reward = 1.0 if done else -0.01

            logger.info(f"[PROG] s*={s_star:.2f}/{s_goal:.2f} ({100.0*s_frac:.1f}%) dist_goal={dist_to_goal:.2f}")

            sat = abs(omega) >= 0.95 * self.ctrl.omega_max
            self._omega_sat_streak = self._omega_sat_streak + 1 if sat else 0
            logger.info(f"[MPC] state=({state[0]:+.2f},{state[1]:+.2f},{state[2]:+.2f}) "
                        f"wp0=({path_xy[0,0]:+.2f},{path_xy[0,1]:+.2f}) "
                        f"v={v:+.3f} ω={omega:+.3f} ok={mpc_dbg.get('solver_ok',False)} "
                        f"iters={mpc_dbg.get('solver_iters',-1)}")

            return self._get_observation(), reward, done, {}
        except Exception:
            logger.exception("Error in step()")
            traceback.print_exc()
            raise
