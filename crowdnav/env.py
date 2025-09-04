# crowdnav/env.py (import block)
import gym, numpy as np, pybullet as p, pybullet_data, traceback
from gym import spaces

from .logger_setup import logger
from .config import TB3, ControlCfg, LidarCfg, Waypointing, WatchdogCfg, MPCWeights
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
omega_max = 2.4
v_max_cap = 0.20

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





class CrowdNavPyBulletEnv(gym.Env):
    def __init__(self, num_peds=0, max_static=8, max_steps=2000, resolution=0.1, seed: int | None = None):
        super().__init__()
        logger.info("CrowdNavPyBulletEnv initializing...")
        self.client = p.connect(p.GUI)
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
        


        self.resolution = resolution
        self.world_size = map_size * MAP_SCALE
        self.half_size  = self.world_size / 2
        self.max_steps  = max_steps
        self.num_peds   = int(num_peds)
        self.max_static = int(max_static)

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
        for w in pts[:max(0, int(vertical_count))]:
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
        for i, (x, y) in enumerate(pts):
            if (i % label_every) != 0:
                continue
            uid = p.addUserDebugText(
                text=str(i),
                textPosition=[float(x), float(y), float(z)],
                textColorRGB=text_rgb,
                textSize=float(text_size),
                lifeTime=0
            )
            self._dbg_path_ids.append(uid)



    def find_free_grid(self, label, avoid=None, min_dist=0.5, max_attempts=200, border_margin=0.2):
        """
        Sample a free world cell for <label>, optionally far from 'avoid'.
        border_margin: keep samples at least this many meters away from world edges.
        """
        
        half = float(self.half_size)
        margin = float(border_margin)
        for _ in range(int(max_attempts)):
            x = self._rng.uniform(-half + margin, half - margin)
            y = self._rng.uniform(-half + margin, half - margin)
            r, c = self.world_to_grid((x, y))
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

        # (optional) clear old debug path items
        for i in getattr(self, "_dbg_path_ids", []):
            try: p.removeUserDebugItem(i)
            except: pass
        self._dbg_path_ids = []

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

        # Draw the path if available
        if replanned:
            self._draw_astar_path(self.path.world_path(),
                                  vertical_count=10, connect=True,
                                  label_every=1, text_size=1.2)

        # Prime HUD (you already throttle updates in step)
        self.hud.follow(self.robot_pos, [
            "MPC HUD",
            "pos: -, - | v,w: -, -",
            f"wp: 0/0 | phase: {self.phase}"
        ])

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
            self.step_count += 1
            self._update_pose()
            state = np.array([*self.robot_pos, self.theta_cont], dtype=float)
            
            replanned = self.path.maybe_replan(self.step_count, self.robot_pos, self.goal_pos)
            if replanned:
                self._draw_astar_path(self.path.world_path(),
                                    vertical_count=10, connect=True,
                                    label_every=1)

            # progress watchdog → v_min_soft + optional speed boost for weights
            dgoal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            from .utils2.watchdog import ProgressWatchdog
            if not hasattr(self, "_wd"):
                self._wd = ProgressWatchdog(self.wdog.progress_eps, self.wdog.stall_window,
                                            self.wdog.vmin_base, self.wdog.vmin_boost, self.wdog.vmin_boost_horizon)
            self._wd.update(dgoal)
            v_min_soft, boosted = self._wd.vmin_soft()

            # (light) global replan
        
            # (light) global replan
            path_xy, _ = self.path.sample_window(self.robot_pos, self.N)

            # Build a heading reference that discourages premature turning:
            # - smooth vs last cycle
            # - freeze beyond first k_strong steps (preview taper)
            theta_ref = _make_theta_ref(path_xy, self._theta_ref_prev, k_strong=3, alpha=0.25)
            self._theta_ref_prev = theta_ref.copy()


            # initial heading to first waypoint
            theta_des0 = np.arctan2(path_xy[0,1] - self.robot_pos[1], path_xy[0,0] - self.robot_pos[0])
            theta_error = _wrap_pi(theta_des0 - self.theta_cont)


            # ALIGN phase
            if self.phase == "ALIGN":
                if self.align.step(theta_error, self.dyn, self.T):
                    self.phase = "TRACK"
                # update pose & return obs+reward like before
                self._update_pose()
                dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
                done = dg < 0.2 or self.step_count >= self.max_steps
                reward = 1.0 if dg < 0.2 else -0.01
                return self._get_observation(), reward, done, {}

            # Build MPC params
            obs_vec = self._get_observation()
            lidar = obs_vec[:self.lcfg.num_rays]
            # derive static candidates from lidar endpoints
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

            # moving obstacles stub
            if self.num_peds > 0:
                obs_traj = np.zeros(2*self.N*self.num_peds)  # (keep same size/shape as your solver expects)
            else:
                obs_traj = np.zeros(2*self.N*self.num_peds)

            speed_w = (self.weights.speed_w_boost if boosted else self.weights.speed_w_base) * 0.6

            weights = np.array([
                18.0,                               # w_track  : ↑ glue to path (lateral/waypoint)
                self.weights.w_goal * 1.5,          # w_goal   : modest terminal pull
                max(0.25, self.weights.w_smooth*2), # w_smooth : ↑ discourages sharp steering
                max(0.5,  self.weights.w_obs),      # w_obs    : ↑ keep margin from walls
                max(0.04, speed_w),                 # speed bias
                0.00                                # w_theta  : **zero or very low** → no eager pre-turn
            ], dtype=float)



            # Corner-aware speed limiting (reduce speed into bends)
            k = min(5, len(theta_ref) - 1)
            corneriness = float(np.max(np.abs(np.diff(theta_ref[:k+1])))) if k > 0 else 0.0
            s = 1.0 / (1.0 + 3.0*corneriness)                  # ~1 on straight; drops near sharp turns
            vmax_local = self.v_max * np.clip(0.35 + 0.65*s, 0.35, 1.0)





            # TRACK (MPC)
            v, omega, U, mpc_dbg = self.track.step(
                state, path_xy, theta_ref, obs_traj, static_flat, weights,
                v_min_soft, vmax_local, omega_max, self.dyn, self.T
            )

            
            logger.debug(f"corner={corneriness:.2f} vmax_loc={vmax_local:.2f}")


            
            
            self._update_pose()

            # Waypoint progress (closest / total along *global* path)
            wp_idx, wp_total = self.path.progress(self.robot_pos)
            if (self.step_count % 6) == 0:
                # Update HUD near the robot
                self.hud.follow(self.robot_pos, [
                    # In HUD follow() call inside step():
                    f"pos: ({self.robot_pos[0]:+.2f},{self.robot_pos[1]:+.2f}) θ={self.theta_cont:+.2f} rad",
                    f"cmd: v={v:+.3f} m/s, ω={omega:+.3f} rad/s",
                    f"wp: {wp_idx}/{wp_total} | phase: {self.phase}",
                    f"U(v) min/max: {mpc_dbg.get('U_minmax_v',[None,None])[0]:.2f}/{mpc_dbg.get('U_minmax_v',[None,None])[1]:.2f}",
                    f"solve: ok={mpc_dbg.get('solver_ok',False)} iters={mpc_dbg.get('solver_iters',-1)}"
                ])

            # Console logs (compact but informative)
            logger.info(
                f"[MPC] state=({state[0]:+.2f},{state[1]:+.2f},{state[2]:+.2f}) "
                f"wp0=({path_xy[0,0]:+.2f},{path_xy[0,1]:+.2f}) "
                f"v={v:+.3f} ω={omega:+.3f} wp_prog={wp_idx}/{wp_total} "
                f"ok={mpc_dbg.get('solver_ok',False)} iters={mpc_dbg.get('solver_iters',-1)}"
            )

            # (Optional) dump first two waypoints and first three theta refs at DEBUG level
            logger.debug(
                f"[MPC IN] path_xy0..1={path_xy[:2].tolist()} "
                f"theta_ref0..2={theta_ref[:3].tolist() if len(theta_ref)>=3 else theta_ref.tolist()} "
                f"weights={weights.tolist()} vmin_soft={v_min_soft:.3f}"
            )



            # read pose
            self._update_pose()

            # reward/done
            dist_to_goal = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            done = dist_to_goal < 0.2 or self.step_count >= self.max_steps
            reward = 1.0 if dist_to_goal < 0.2 else -0.01
            return self._get_observation(), reward, done, {}

        except Exception:
            logger.exception("Error in step()")
            traceback.print_exc()
            raise
