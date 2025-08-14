import pybullet as p
import pybullet_data
import numpy as np
import time
import gym
import matplotlib.pyplot as plt

from gym import spaces
from .planner import a_star
from .mpc import build_mpc_solver_random_obs
from .world import create_world
from .pedestrian import create_random_pedestrian, update_pedestrian
from .utils import get_random_position, find_free_position, update_astar_path
from .utils import MAP_SCALE, map_size, TURTLEBOT_RADIUS
from .logger_setup import logger


K_TURN   = 2        # number of pure-turn steps (2–3 is typical)
v_go_min = 0.05     # minimum forward speed once we start moving
omega_max = 2.4
v_max     = 0.20


import traceback


class CrowdNavPyBulletEnv(gym.Env):
    
    """Main Gym environment for CrowdNav with PyBullet."""
    def __init__(self, num_peds=0, max_static=1, max_steps=2000, resolution=0.1):
        super().__init__()
        try:
            logger.info("CrowdNavPyBulletEnv initializing...")
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Environment parameters
            self.resolution = resolution
            self.world_size = map_size * MAP_SCALE   # total width in meters
            
            # In __init__(), after you set resolution, world_size, etc.:
            self.num_rays  = 32                    # how many beams per scan
            self.fov       = 270 * np.pi / 180     # 270° field of view
            self.max_range = 8.0                   # maximum distance (m)
            self.lidar_height = 0.175
            self.half_size  = self.world_size / 2    # half‑width in meters
            self.max_steps = max_steps
            # Still in __init__(), right after the above:
            angles = np.linspace(-self.fov/2, +self.fov/2, self.num_rays)
            self.lidar_dirs = np.stack([
                np.cos(angles),                     # x components
                np.sin(angles),                     # y components
                np.zeros_like(angles)               # z = 0 (planar)
            ], axis=1)  # shape = (num_rays, 3)
            
            
            
            self.debug_window = 20
            self.debug_rows = []   # list of dicts; flushed to disk after 50 rows

            # self.static_obs = static_obs
            self.step_count = 0
            self.robot_pos = np.array([1.0, 1.0])  # Initial placeholder, will be randomized in reset
            self.theta = 0.0
            self.goal_pos = np.array([4.0, 4.0])  # Initial suggestion, will be validated
            self.num_peds = num_peds
            self.max_static = 8
            self.peds = []
            self.static_obs = []
            
            


            # Initialize world and robot
            self.robot_id, self.left_wheel_joint_id, self.right_wheel_joint_id, self.grid, self.static_obs = create_world(
                self.client, resolution=self.resolution
            )
            
            self.grid_size = self.grid.shape
            # print(f"Grid Shape: {self.grid_size}")

            # # MPC setup
            # self.solver, self.f_dyn, self.T, self.N = build_mpc_solver_random_obs(max_obs=0, max_static=0)
            
                
            self.solver, self.f_dyn, self.T, self.N = build_mpc_solver_random_obs(
                max_obs=self.num_peds,
                max_static=self.max_static
            )
              
            #We are getting start and goal poses inside of init, this is being done to initialize the environment?
            goal_idx    = self.find_free_grid(label="goal")
            self.goal_pos = self.grid_to_world(goal_idx)
            
            # Replace any old hard-coded dims:
            obs_dim = self.num_rays + 2 + 1       # ranges + goal_dir(x,y) + goal_dist
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(obs_dim,), dtype=np.float32
            )
            
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            
            # --- Heading control & waypointing ---
            self.lookahead_wp = 2           # how many waypoints to look ahead from the closest
            self.turn_eps_enter = 0.25      # rad (~14°): enter TURN mode if heading error ≥ this
            self.turn_eps_exit  = 0.10      # rad (~6°):  exit TURN mode if heading error ≤ this
            self.omega_turn_max = 0.8       # rad/s cap for turn-in-place (prevents wild spinning)
            self.mode = "GO"                # TURN | GO


                        # --- Phases ---
            self.phase = "ALIGN"        # ALIGN -> TRACK (MPC)
            self.align_eps = 0.15       # ~8.6° target alignment before switching
            self.align_hold = 10         # need N consecutive aligned steps to switch
            self._align_ctr = 0
            self.kp_align = 1.2         # ω = kp * θ_err, clipped by omega_max
            self.omega_align_max = min(self.omega_turn_max, omega_max)  # realism



            # --- TurtleBot3 Burger realistic limits ---
            self.motor_force = 1.8          # N·m per wheel (1.8–2.0 realistic)
            self.max_wheel_omega = 6.30      # rad/s  (~61 rpm @ 12V)
            self.v_max_hw = 0.033 * self.max_wheel_omega  # ≈ 0.208 m/s
            self.v_max = min(v_max, self.v_max_hw)        # ensure v_max ≤ hardware
            
            
            # --- Progress / stall watchdog params ---
            self.progress_eps = 1e-3        # meters of goal improvement to count as progress
            self.stall_window = 15          # consecutive steps without progress → "stall"
            self.stall_ctr = 0              # counts no-progress steps
            self.vmin_base = 0.03           # baseline soft forward lower bound
            self.vmin_boost = 0.08          # boosted v_min when stalling
            self.vmin_boost_horizon = 30    # how many steps to keep boost
            self.vmin_boost_ctr = 0         # countdown for boost
            self._dg_prev = None            # previous distance to goal (for progress)




                        
            
            logger.info(f"World ready | grid shape: {self.grid.shape}, res: {self.resolution}, N={self.N}, T={self.T}")



        except Exception as e:
            print(f"Error in __init__: {e}")
            logger.exception("Error in __init__")
            traceback.print_exc()
            raise
        


        # ----------------------------------------------------------------
    # Helpers for a single canonical coord convention:
    #   grid indices = (row, col)
    #   world coords   = (x,   y) in meters
    # ----------------------------------------------------------------
    def world_to_grid(self, pos):
        """(x,y) [m] → (row,col) on self.grid."""
        c = int((pos[0] + self.half_size) / self.resolution)
        r = int((pos[1] + self.half_size) / self.resolution)
        return r, c
    def grid_to_world(self, idx):
        """(row,col) → (x,y) [m] at the *center* of that cell."""
        x = (idx[1] + 0.5) * self.resolution - self.half_size
        y = (idx[0] + 0.5) * self.resolution - self.half_size
        return np.array([x, y])

        

    def find_free_grid(self, label, avoid=None, min_dist=0.5, max_attempts=100):
        """Sample a free (row,col) within the scaled world, respecting a minimum distance from `avoid`."""
        for _ in range(max_attempts):
            # 1) sample in the true bounds ±half_size
            x = np.random.uniform(-self.half_size, self.half_size)
            y = np.random.uniform(-self.half_size, self.half_size)
            # 2) convert to grid and clamp to [0..grid.shape-1]
            r, c = self.world_to_grid((x, y))
            r = np.clip(r, 0, self.grid.shape[0] - 1)
            c = np.clip(c, 0, self.grid.shape[1] - 1)
            # 3) accept if free and far enough from avoid
            if self.grid[r, c] != 0:
                continue
            world = self.grid_to_world((r, c))
            if avoid is None or np.linalg.norm(world - avoid) >= min_dist:
                return (r, c)
        raise ValueError(f"Could not find free {label} cell after {max_attempts} tries")



    def _plot_environment_debug(self):
        import matplotlib.pyplot as plt
        plt.ioff()  # Prevents multiple plot windows in quick succession

        plt.figure(figsize=(6, 6))
        plt.imshow(self.grid.T, origin='lower', cmap='gray_r')  # Show walls and free space

        # Plot A* path if available
        if self.global_path_idx is not None and len(self.global_path_idx) > 0:
            path = np.array(self.global_path_idx)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='A* Path')  # path is in (row, col)

            # Mark start and goal on top of the path for clarity
            plt.plot(path[0, 1], path[0, 0], 'go', label='Start')
            plt.plot(path[-1, 1], path[-1, 0], 'ro', label='Goal')
        else:
            print("Warning: A* path is empty or not initialized.")

        plt.legend()
        plt.title("A* dadaPath on Occupancy Grid")
        plt.tight_layout()
        plt.grid(False)
        plt.show()
        plt.close()



    def reset(self, visualize=False):
        """Reset the environment to a fixed start/goal for testing."""
        self.step_count = 0
        self.theta = 0.0
        
        
        self.phase = "ALIGN"
        self._align_ctr = 0
                

        # 1) the *screen* click was at (x=88, y=18)
        #    so grid indices = (row=y, col=x):
        manual_start = (18, 88)
        manual_goal  = (14, 22)


        # ─── 2) SANITY CHECKS (1 means obstacle):
        if self.grid[manual_start] == 1:
            raise ValueError(f"Manual start {manual_start} lies inside an obstacle!")
        if self.grid[manual_goal] == 1:
            raise ValueError(f"Manual goal  {manual_goal}  lies inside an obstacle!")

        # ─── 3) STORE and convert once:
        self.start_idx = manual_start
        self.goal_idx  = manual_goal
        self.robot_pos = self.grid_to_world(self.start_idx)
        self.goal_pos  = self.grid_to_world(self.goal_idx)

        # ─── 4) Place robot/goal in PyBullet:
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [*self.robot_pos, 0.008],
            p.getQuaternionFromEuler([0, 0, self.theta])
        )

        p.addUserDebugLine(
            [*self.goal_pos, 0],
            [*self.goal_pos, 10],
            [0,1,0], lineWidth=3, lifeTime=0
        )

        # ─── 5) Plan A* and show grid:
        self.global_path_idx = update_astar_path(
            self.robot_pos,
            self.goal_pos,
            self.grid,
            self.resolution
        )
        
        
        
        # Draw vertical lines for only the first 5 A* path waypoints
        if self.global_path_idx:
            for idx in self.global_path_idx[:5]:
                world_pos = self.grid_to_world(idx)  # Convert (row, col) to (x, y)
                p.addUserDebugLine(
                    [world_pos[0], world_pos[1], 0],   # Start at z=0
                    [world_pos[0], world_pos[1], 10],  # End at z=10
                    [0, 0, 1],                         # Blue color for waypoints
                    lineWidth=2,
                    lifeTime=0                         # Persistent lines
                )

            
        
        #To show the occupancy grid for 
        # self.show_occupancy_grid(self.grid, self.global_path_idx)
        
        
        try:
            for link in [self.left_wheel_joint_id, self.right_wheel_joint_id]:
                p.changeDynamics(self.robot_id, linkIndex=link,
                                lateralFriction=0.6,
                                rollingFriction=0.0,
                                spinningFriction=0.0,
                                linearDamping=0.02,
                                angularDamping=0.02)
        except Exception as e:
            logger.debug(f"changeDynamics skipped: {e}")

        
        
        logger.info("=== reset(): new episode ===")
        logger.debug(f"start_idx={manual_start}, goal_idx={manual_goal}")
        logger.debug(f"start_world={self.grid_to_world(self.start_idx)}, goal_world={self.grid_to_world(self.goal_idx)}")

        return self._get_observation()


    def _plot_astar_path(self):
            """Plot the A* path on the occupancy grid, centered correctly."""
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8,8))
            # Show grid with (0,0) at bottom‑left
            plt.imshow(self.grid, origin='lower', cmap='gray_r')

            if self.global_path_idx:
                path = np.array(self.global_path_idx)   # shape (N,2): (row, col)
                rows, cols = path[:,0], path[:,1]

                # plot through the *centers* of the cells
                plt.plot(cols + 0.5, rows + 0.5, 'b-', linewidth=2, label='A* Path')
                plt.plot(cols[0] + 0.5, rows[0] + 0.5, 'go', label='Start')
                plt.plot(cols[-1] + 0.5, rows[-1] + 0.5, 'ro', label='Goal')

            plt.legend()
            plt.title("A* Path on Occupancy Grid")
            plt.grid(False)
            plt.show()



    def step(self, action):
        """Execute one step in the environment."""
        N = self.N
        try:
            self.step_count += 1
            state = np.array([*self.robot_pos, self.theta])
            logger.info(f"[Step {self.step_count}] pose=({self.robot_pos[0]:.3f},{self.robot_pos[1]:.3f}), theta={self.theta:.3f}")

            # --- Progress & stall bookkeeping ---
            dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            if self._dg_prev is not None:
                if (self._dg_prev - dg) < self.progress_eps:
                    self.stall_ctr += 1
                else:
                    self.stall_ctr = 0
            else:
                self.stall_ctr = 0
            self._dg_prev = dg

            # Trigger a temporary forward bias if stalling
            if self.stall_ctr >= self.stall_window and self.vmin_boost_ctr == 0 and dg > 0.3:
                logger.warning("[WATCHDOG] No progress for a while → boosting v_min and speed cost")
                self.vmin_boost_ctr = self.vmin_boost_horizon

            # Soft forward bound to use this step
            v_min_soft = self.vmin_boost if self.vmin_boost_ctr > 0 else self.vmin_base
            if self.vmin_boost_ctr > 0:
                self.vmin_boost_ctr -= 1

            # Light periodic/global replan (and also when stalled)
            if (self.step_count % 10 == 0) or (self.stall_ctr >= self.stall_window):
                try:
                    self.global_path_idx = update_astar_path(
                        self.robot_pos, self.goal_pos, self.grid, self.resolution
                    )
                    logger.debug(f"[REPLAN] global_path_idx length={len(self.global_path_idx)}")
                except Exception as e:
                    logger.warning(f"[REPLAN] failed: {e}")




            # Prepare MPC inputs
            if not self.global_path_idx:
                print("No valid A* path, using goal position as fallback")
                half_size = self.half_size
                sampled = [((self.goal_pos + half_size) / self.resolution).astype(int)] * self.N
            else:
                # Convert all waypoints to world coordinates
                world_path = [self.grid_to_world(idx) for idx in self.global_path_idx]
                distances = [np.linalg.norm(self.robot_pos - np.array(pos)) for pos in world_path]
                closest_idx = np.argmin(distances)
                start_idx = min(closest_idx + self.lookahead_wp, len(self.global_path_idx) - 1)

                if start_idx + self.N <= len(self.global_path_idx):
                    sampled = self.global_path_idx[start_idx:start_idx + self.N]
                else:
                    sampled = self.global_path_idx[start_idx:] + [self.global_path_idx[-1]] * (self.N - (len(self.global_path_idx) - start_idx))

            
                logger.debug(f"closest_idx={closest_idx}, closest_wp={world_path[closest_idx]}")
                sampled_world = [self.grid_to_world(idx) for idx in sampled]
                #print(f"[DEBUG] Sampled waypoints: {sampled_world[:3]}")
                logger.debug(f"Sampled waypoints (first 3): {sampled_world[:3]}")
                
                
            # After you compute `sampled` (list of (row, col)):
            path_xy = np.array([self.grid_to_world(idx) for idx in sampled])  # (N,2)

            # --- REMOVE leading waypoint(s) that are basically at the current pose ---
            tol = 2.0 * self.resolution
            mask = np.linalg.norm(path_xy - self.robot_pos[None, :], axis=1) > tol
            if not np.any(mask):
                # fallback: use goal if everything was filtered
                path_xy = np.array([self.goal_pos])
            else:
                path_xy = path_xy[mask]

            # pad back to N by repeating the last waypoint if needed
            if len(path_xy) < self.N:
                pad = np.repeat(path_xy[-1][None, :], self.N - len(path_xy), axis=0)
                path_xy = np.vstack([path_xy, pad])

            # Recompute flattened path & heading refs
            path   = path_xy.flatten()
            deltas = path_xy[1:] - path_xy[:-1]
            raw_ref = np.arctan2(deltas[:,1], deltas[:,0])
            theta_ref = np.concatenate([raw_ref, raw_ref[-1:]])

                            
            
            logger.debug(f"path_xy[0]={path_xy[0]}, path_xy[1]={path_xy[1] if len(path_xy)>1 else 'n/a'}")
            logger.debug(f"theta_ref[0:3]={theta_ref[:3]}")


            first_wp   = path_xy[0]
            theta_des0 = np.arctan2(first_wp[1] - self.robot_pos[1], first_wp[0] - self.robot_pos[0])

            theta_error = theta_des0 - self.theta
            theta_error = (theta_error + np.pi) % (2*np.pi) - np.pi

            logger.info(f"first_wp={first_wp}, theta_des0={theta_des0:.3f}, theta={self.theta:.3f}, theta_err={theta_error:.3f}")

            # ---------- ALIGN PHASE (no MPC) ----------
            if self.phase == "ALIGN":
                # proportional turn-in-place, v=0
                omega_cmd = float(np.clip(self.kp_align * theta_error, -self.omega_align_max, self.omega_align_max))
                v_cmd = 0.0

                # convert to wheel rates (clamped to TB3 caps you already have)
                wheel_radius = 0.033
                wheel_base   = 0.160
                w_left  = (v_cmd - omega_cmd * wheel_base / 2.0) / wheel_radius
                w_right = (v_cmd + omega_cmd * wheel_base / 2.0) / wheel_radius
                w_left  = float(np.clip(w_left,  -self.max_wheel_omega, self.max_wheel_omega))
                w_right = float(np.clip(w_right, -self.max_wheel_omega, self.max_wheel_omega))

                p.setJointMotorControl2(self.robot_id, self.left_wheel_joint_id,
                                        controlMode=p.VELOCITY_CONTROL, targetVelocity=w_left, force=self.motor_force)
                p.setJointMotorControl2(self.robot_id, self.right_wheel_joint_id,
                                        controlMode=p.VELOCITY_CONTROL, targetVelocity=w_right, force=self.motor_force)

                # advance physics for exactly self.T
                fixed = p.getPhysicsEngineParameters()['fixedTimeStep']
                steps = max(1, int(np.ceil(self.T / fixed)))
                for _ in range(steps):
                    p.stepSimulation()

                # update pose
                pos, orn = p.getBasePositionAndOrientation(self.robot_id)
                self.robot_pos = np.array(pos[:2])
                self.theta = p.getEulerFromQuaternion(orn)[2]
                self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi

                # switch to TRACK only after being aligned for a few consecutive steps
                if abs(theta_error) < self.align_eps:
                    self._align_ctr += 1
                else:
                    self._align_ctr = 0
                if self._align_ctr >= self.align_hold:
                    self.phase = "TRACK"
                    logger.info("[PHASE] ALIGN → TRACK")

                # log and return like a normal step (no reward shaping change)
                dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
                done = dg < 0.2 or self.step_count >= self.max_steps
                reward = 1.0 if dg < 0.2 else -0.01
                # (optional) debug line
                p.addUserDebugLine([self.robot_pos[0], self.robot_pos[1], 0],
                                [self.robot_pos[0], self.robot_pos[1], 0.15], [1,0,0], lineWidth=2, lifeTime=0.3)
                logger.info(f"[ALIGN] |θ_err|={abs(theta_error):.3f} ω={omega_cmd:.3f} ctr={self._align_ctr}/{self.align_hold}")
                return self._get_observation(), reward, done, {}
            # ---------- end ALIGN ----------





            half_size = self.half_size

            

            if self.num_peds > 0:
                obs_traj = []
                for ped in self.peds:
                    direction = ped['goal'] - ped['pos']
                    direction /= np.linalg.norm(direction) + 1e-6
                    for k in range(self.N):
                        obs_traj.append(ped['pos'] + 0.02 * k * direction)
                obs_traj = np.array(obs_traj).flatten()
            else:
                obs_traj = np.zeros(2 * self.N * self.num_peds)

            # Check collisions
            contact_points = p.getContactPoints()

            
            lbx, ubx = [], []
            
            # 1) States first
            lbx = [-1e20] * 3 * (self.N + 1)
            ubx = [ 1e20] * 3 * (self.N + 1)

            # 2) Controls next — tiny forward bias, still realistic
            for k in range(self.N):
                lbx += [v_min_soft, -omega_max]
                ubx += [self.v_max,   omega_max]

            
            
            exp = self.solver.size_in('x0')
            exp_n = exp[0] if isinstance(exp, tuple) else int(exp)
            assert len(lbx) == exp_n and len(ubx) == exp_n, (len(lbx), exp)
            
            logger.debug(f"x0 expected len={exp_n}, lbx/ubx len={len(lbx)}")
            logger.debug(f"First 4 v-bounds: {[(lbx[2*k], ubx[2*k]) for k in range(min(4, self.N))]}")

            
            print("lbx len:", len(lbx), "ubx len:", len(ubx))
            print("solver expects x0 of length:", self.solver.size_in('x0'))
            print(f"your lbx has length {len(lbx)}, solver expects {self.solver.size_in('x0')}")
            
            
            nX = 3*(self.N + 1)
            nU = 2*self.N
            u0 = np.zeros((nX + nU, 1))  # [all X | all U]

            # Small forward velocity + a gentle first-step turn guess
            omega_guess = np.clip(theta_error / self.T, -omega_max, omega_max)
            for k in range(self.N):
                v_k = max(0.10, v_go_min)   # small but nonzero forward
                w_k = omega_guess if k == 0 else 0.0
                u0[nX + 2*k + 0, 0] = v_k
                u0[nX + 2*k + 1, 0] = w_k

                    
                    
            

            logger.debug(f"omega_guess={omega_guess:.3f}, v_go_min={v_go_min:.3f}")
            logger.debug(f"First 4 u0 (v,ω): {[(float(u0[2*k]), float(u0[2*k+1])) for k in range(min(4, self.N))]}")


            g_len = self.N * 3 + 3

            pos, quat = p.getBasePositionAndOrientation(self.robot_id)
            pos = np.array(pos)
            pos[2] = self.lidar_height
            R = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            origins = np.repeat(pos[None, :], self.num_rays, axis=0)
            dirs_world = (R @ self.lidar_dirs.T).T
            
            obs = self._get_observation()
            lidar = obs[:self.num_rays]
            hit_pts = origins + (lidar[:, None] * self.max_range) * dirs_world
            
            valid = lidar < 1.0
            pts2d = hit_pts[valid, :2]
            dists = np.linalg.norm(pts2d - self.robot_pos, axis=1)
            order = np.argsort(dists)
            
            closest = pts2d[order][:self.max_static]
            logger.debug(f"num_valid_hits={int(valid.sum())}, using_static_pts={closest.shape[0]}")

            
            
            static_flat = closest.flatten()
            if static_flat.size < 2 * self.max_static:
                pad = np.full(2 * self.max_static - static_flat.size, 1e6, dtype=float)  # “no obstacle”
                static_flat = np.concatenate([static_flat, pad])

                
                
                
                

            state = np.array([*self.robot_pos, self.theta])
            
            
            w_goal   = 5.0
            w_smooth = 0.10        # small smoothing helps avoid dithering
            w_obs    = 0.5
            w_theta  = 0.1
            speed_w  = 0.3 if self.vmin_boost_ctr > 0 else 0.1  # stronger bias only while boosted

            weights = np.array([
                w_goal * 10.0,
                w_smooth * 1.0,
                w_obs * 1.0,
                speed_w,
                w_theta * 2.0
            ])

            
            P = np.concatenate([state, path, theta_ref, obs_traj, static_flat, weights])
            P = P.reshape((-1, 1))
            assert P.shape == self.solver.size_in('p')
            logger.debug(f"P path refs (first 6 coords): {P[3:3+6].T}")
            
            # sanity-check parameter budgeting
            n_state    = 3
            n_path     = 2 * self.N
            n_thref    = self.N
            n_obs_traj = 2 * self.N * self.num_peds
            n_static   = 2 * self.max_static
            n_weights  = 5
            total_p    = n_state + n_path + n_thref + n_obs_traj + n_static + n_weights

            logger.debug(f"|p| actual={P.size}, size_in('p')={self.solver.size_in('p')}")

            logger.debug(f"P path refs (first 3 waypoints): {path_xy[:3]}")
                                
                                
            dbg = {}

            # basic geometry
            dbg["step"] = int(self.step_count)
            dbg["phase"] = self.phase
            dbg["theta"] = float(self.theta)
            dbg["theta_des0"] = float(theta_des0)
            dbg["theta_err"] = float(theta_error)

            # path refs
            dbg["closest_idx"] = int(closest_idx) if 'closest_idx' in locals() else -1
            dbg["lookahead_wp"] = int(self.lookahead_wp)
            dbg["path_xy0"] = path_xy[0].tolist()
            if len(path_xy) > 1:
                dbg["path_xy1"] = path_xy[1].tolist()

            # lidar stats
            dbg["lidar_min"] = float(lidar.min())
            dbg["lidar_mean"] = float(lidar.mean())
            dbg["num_hits"] = int((lidar < 1.0).sum())
            dbg["num_static_used"] = int(closest.shape[0]) if 'closest' in locals() else 0

            # bounds & init
            dbg["v_min_soft"] = float(v_min_soft) if "v_min_soft" in locals() else 0.0
            dbg["v_max_used"] = float(self.v_max)
            dbg["u0_first"] = [float(u0[-2*self.N + 0]), float(u0[-2*self.N + 1])]  # (v0, ω0) as you pack [X|U]

            # weights actually used
            dbg["w_goal"] = float(w_goal)
            dbg["w_smooth"] = float(w_smooth)
            dbg["w_obs"] = float(w_obs)
            dbg["w_theta"] = float(w_theta)
            dbg["speed_w"] = float(speed_w)

            # goal progress
            dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            dbg["d_goal"] = dg

                                
                                
            sol = self.solver(
                x0=u0,
                p=P,
                lbg=[0] * g_len,
                ubg=[0] * g_len,
                lbx=lbx,
                ubx=ubx
            )
            
           
            # --- extract controls from the tail (states-first, controls-after) ---
            z  = sol['x'].full().flatten()
            nX = 3 * (self.N + 1)          # all states: [x0,y0,th0, ..., xN,yN,thN]
            nU = 2 * self.N                # all controls: [v0,w0, v1,w1, ... vN-1,wN-1]

            U = z[nX : nX + nU].reshape(self.N, 2)   # shape (N,2), columns = (v, ω)
            v, omega = U[0]
            u_opt = U
            
            
            
            
            
            #Sanity checks
            v_lb, v_ub = (0.0, v_max)
            assert v_lb - 1e-9 <= v <= v_ub + 1e-9, f"v out of bounds: {v:.4f}"
            assert -omega_max - 1e-9 <= omega <= omega_max + 1e-9, f"ω out of bounds: {omega:.4f}"


            
            logger.info(f"[CTRL] v={v:.3f}, ω={omega:.3f}")
            
            for i in range(self.N):
                v_i, omega_i = u_opt[i]
                logger.debug(f"[H{i:02d}] v={v_i:.3f}, ω={omega_i:.3f}")
                
                
                
            v_min_h = float(u_opt[:,0].min())
            v_max_h = float(u_opt[:,0].max())
            logger.debug(f"[H] v_min={v_min_h:.3f}, v_max={v_max_h:.3f}, omega0={omega:.3f}")

            # Progress monitor
            prev = getattr(self, "_dg_prev", None)
            dg = float(np.linalg.norm(self.goal_pos - self.robot_pos))
            if prev is not None:
                logger.debug(f"[PROGRESS] d_goal: {prev:.3f} -> {dg:.3f} (Δ {prev-dg:+.3f})")
            self._dg_prev = dg


            expected_shape = self.solver.size_in('p')
            logger.debug(f"P.shape={P.shape}, solver expects {expected_shape}")
            assert P.shape == expected_shape, f"P-shape mismatch: got {P.shape}, wants {expected_shape}"


            # Apply control to robot
            wheel_radius = 0.033
            wheel_base   = 0.160

            # Angular wheel speeds (rad/s)
            w_left  = (v - omega * wheel_base / 2.0) / wheel_radius
            w_right = (v + omega * wheel_base / 2.0) / wheel_radius

            # Clamp to hardware
            w_left  = float(np.clip(w_left,  -self.max_wheel_omega, self.max_wheel_omega))
            w_right = float(np.clip(w_right, -self.max_wheel_omega, self.max_wheel_omega))

            p.setJointMotorControl2(self.robot_id, self.left_wheel_joint_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=w_left,
                                    force=self.motor_force)
            p.setJointMotorControl2(self.robot_id, self.right_wheel_joint_id,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=w_right,
                                    force=self.motor_force)


            # Advance simulation by self.T seconds
            resetfixed_time_step = p.getPhysicsEngineParameters()['fixedTimeStep']
            num_substeps = max(1, int(np.ceil(self.T / resetfixed_time_step)))
            for _ in range(num_substeps):
                p.stepSimulation()
                
                
            try:
                l_vel = p.getJointState(self.robot_id, self.left_wheel_joint_id)[1]   # rad/s
                r_vel = p.getJointState(self.robot_id, self.right_wheel_joint_id)[1]
                logger.debug(f"[WHEELS] tgt=({w_left:.2f},{w_right:.2f}) act=({l_vel:.2f},{r_vel:.2f})")
            except Exception:
                pass



            # Retrieve new position and orientation
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            self.robot_pos = np.array(pos[:2])
            self.theta = p.getEulerFromQuaternion(orn)[2]
            self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
            
            
            
            # --- STUCK DETECTOR ---
            if not hasattr(self, "_last_pos"):
                self._last_pos = self.robot_pos.copy()
            moved = float(np.linalg.norm(self.robot_pos - self._last_pos))
            if moved < 1e-3:
                logger.warning("[STUCK?] Δpos < 1 mm this step")
            self._last_pos = self.robot_pos.copy()


            # # Debug print to verify state update
            # print(f"[Step {self.step_count}] Robot pos: {self.robot_pos}, theta: {self.theta:.3f}")
            
            
            # logger.debug(f"wheel_cmds: vL={v_left:.3f}, vR={v_right:.3f}")
            # After physics step and reading pose:
            logger.info(f"[STATE] pos=({self.robot_pos[0]:.3f},{self.robot_pos[1]:.3f}), theta={self.theta:.3f}")

            # Add debug line for robot (red)
            # p.addUserDebugLine([self.robot_pos[0], self.robot_pos[1], 0], [self.robot_pos[0], self.robot_pos[1], 10], [1, 0, 0], lineWidth=3, lifeTime=0)


            dbg["u_opt_0"] = [float(u_opt[0,0]), float(u_opt[0,1])]
            dbg["u_opt_1"] = [float(u_opt[1,0]), float(u_opt[1,1])] if self.N > 1 else None
            dbg["u_opt_minmax_v"] = [float(u_opt[:,0].min()), float(u_opt[:,0].max())]

            # wheel targets & actuals
            dbg["w_left_tgt"] = float(w_left)
            dbg["w_right_tgt"] = float(w_right)
            try:
                l_vel = p.getJointState(self.robot_id, self.left_wheel_joint_id)[1]
                r_vel = p.getJointState(self.robot_id, self.right_wheel_joint_id)[1]
                dbg["w_left_act"] = float(l_vel)
                dbg["w_right_act"] = float(r_vel)
            except Exception:
                pass

            # solver stats (if available)
            try:
                st = self.solver.stats()
                dbg["solver_ok"] = bool(st.get("success", False))
                dbg["solver_iters"] = int(st.get("iter_count", -1))
                dbg["solver_status"] = str(st.get("return_status", ""))
            except Exception:
                pass

            # stash and flush
            if len(self.debug_rows) < self.debug_window:
                self.debug_rows.append(dbg)
                if len(self.debug_rows) == self.debug_window:
                    import json, time, os
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = f"mpc_debug_first{self.debug_window}_{ts}.jsonl"
                    with open(path, "w") as f:
                        for row in self.debug_rows:
                            f.write(json.dumps(row) + "\n")
                    logger.warning(f"[DEBUG] wrote first {self.debug_window} MPC rows to {os.path.abspath(path)}")





            # Compute reward and done condition
            dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
            done = dist_to_goal < 0.2 or self.step_count >= self.max_steps
            reward = 1.0 if dist_to_goal < 0.2 else -0.01
            return self._get_observation(), reward, done, {}
        
            
        
        
        
        
        
        



        except Exception as e:
            logger.exception("Error in step()")
            traceback.print_exc()
            raise
    
    def _get_observation(self):
        # 1) Robot pose in world
        pos, quat = p.getBasePositionAndOrientation(self.robot_id)
        pos = np.array(pos)                                 # (x,y,z)
        pos[2] = self.lidar_height
        R   = np.array(p.getMatrixFromQuaternion(quat))\
                .reshape(3,3)                               # rotation matrix

        # 2) Beam start/end points
        origins    = np.repeat(pos[None,:], self.num_rays, axis=0)          # (num_rays,3)
        dirs_world = (R @ self.lidar_dirs.T).T                              # (num_rays,3)
        ends       = origins + dirs_world * self.max_range                  # (num_rays,3)

        # 3) Cast rays in batch
        results = p.rayTestBatch(
            rayFromPositions=origins.tolist(),
            rayToPositions  =ends.tolist()
        )
        
        
        #LIDAR VISUALIZATION( MAKES EVERYTHING LAG )
        # for ori, end in zip(origins, ends):
        #     p.addUserDebugLine(ori, end, [1,0,0], lifeTime=0.1)

        # 4) Extract normalized distances
        lidar = np.array([hit[2] for hit in results], dtype=np.float32)
        # hit[2] = fraction along ray (1.0 means no hit within max_range)

        # 5) Compute goal info
        goal_vec  = self.goal_pos - pos[:2]
        goal_dist = np.linalg.norm(goal_vec) + 1e-6
        goal_dir  = goal_vec / goal_dist

        # 6) Return full observation
        return np.concatenate([lidar, goal_dir, [goal_dist]])





    def show_occupancy_grid(self, grid, path=None):
        """Your main debug plot: grid + ★/✕ + optional A* path."""
        plt.figure(figsize=(6,6))
        plt.imshow(grid, origin='lower', cmap='gray_r')

        # center your manual start/goal markers too
        sr, sc = self.start_idx
        gr, gc = self.goal_idx
        plt.scatter([sc + 0.5], [sr + 0.5], c='g', s=100, marker='*', label='Start')
        plt.scatter([gc + 0.5], [gr + 0.5], c='r', s=100, marker='X', label='Goal')

        if path:
            arr = np.array(path)
            rows, cols = arr[:,0], arr[:,1]
            plt.plot(cols + 0.5, rows + 0.5, 'b-', linewidth=2, label='A* Path')
        else:
            plt.text(0.5, 0.95, "No valid A* path",
                     color='orange',
                     transform=plt.gca().transAxes,
                     ha='center', va='top',
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        plt.legend(loc='upper right')
        plt.title("Occupancy Grid — Start (★), Goal (✕)")
        plt.axis('off')
        plt.show()




if __name__ == "__main__":
    try:
        env = CrowdNavPyBulletEnv()
        logger.info("Environment initialized successfully")
        obs = env.reset(visualize=True)
        logger.info("Reset successful")
        while True:
            action = np.array([1.0, 0.1, 0.1])
            obs, reward, done, _ = env.step(action)

            contacts = p.getContactPoints(bodyA=env.robot_id)
            if contacts:
                logger.warning(f"Collision: {len(contacts)} contact(s)")
                for c in contacts:
                    logger.warning(
                        f"  bodyB={c[2]} linkA={c[3]} linkB={c[4]} "
                        f"pos={np.round(c[5],3)}, normal={np.round(c[7],3)}, depth={c[8]:.4f}"
                    )
            else:
                logger.debug("No collision detected.")

            if done:
                logger.info("Episode finished.")
                break
        logger.info("Episode finished.")
    except Exception:
        logger.exception("Main loop error")
        p.disconnect()
