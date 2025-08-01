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


import traceback


class CrowdNavPyBulletEnv(gym.Env):
    
    """Main Gym environment for CrowdNav with PyBullet."""
    def __init__(self, num_peds=0, max_static=1, max_steps=200, resolution=0.1):
        super().__init__()
        try:
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Environment parameters
            self.resolution = resolution
            self.world_size = map_size * MAP_SCALE   # total width in meters
            
            # In __init__(), after you set resolution, world_size, etc.:
            self.num_rays  = 32                    # how many beams per scan
            self.fov       = 270 * np.pi / 180     # 270° field of view
            self.max_range = 5.0                   # maximum distance (m)
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

            # self.static_obs = static_obs
            self.step_count = 0
            self.robot_pos = np.array([1.0, 1.0])  # Initial placeholder, will be randomized in reset
            self.theta = 0.0
            self.goal_pos = np.array([4.0, 4.0])  # Initial suggestion, will be validated
            self.num_peds = num_peds
            self.max_static = max_static 
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


        except Exception as e:
            print(f"Error in __init__: {e}")
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
        self.show_occupancy_grid(self.grid, self.global_path_idx)

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
        try:
            p.stepSimulation()
            self.step_count += 1
            w_goal, w_smooth, w_obs = np.clip(action, 0.0, 1.0)
            state = np.array([*self.robot_pos, self.theta])

            # Prepare MPC inputs
            if not self.global_path_idx:
                print("No valid A* path, using goal position as fallback")
                half_size = self.half_size
                sampled = [((self.goal_pos + half_size) / self.resolution).astype(int)] * self.N

            elif len(self.global_path_idx) < self.N:
                sampled = self.global_path_idx + [self.global_path_idx[-1]] * (self.N - len(self.global_path_idx))
            else:
                sampled = np.linspace(0, len(self.global_path_idx) - 1, self.N).astype(int)
                sampled = [self.global_path_idx[i] for i in sampled]
            # Convert grid indices to unscaled world coordinates
            half_size = self.half_size
            path = [np.array(idx) * self.resolution - half_size for idx in sampled]
            path = np.array(path).flatten()

            obs_traj = []
            for ped in self.peds:
                direction = ped['goal'] - ped['pos']
                direction /= np.linalg.norm(direction) + 1e-6
                for k in range(1, self.N + 1):
                    obs_traj.append(ped['pos'] + 0.02 * k * direction)
            obs_traj = np.array(obs_traj).flatten()

            static_obs_nearby = [obs for obs in self.static_obs if np.linalg.norm(obs - self.robot_pos) < 3.0]
            static_flat = np.array(static_obs_nearby[:self.max_static]).flatten()
            if len(static_flat) < 2 * self.max_static:
                padding = np.zeros(2 * self.max_static - len(static_flat))
                static_flat = np.concatenate([static_flat, padding])

            # Check collisions
            contact_points = p.getContactPoints()
            # if contact_points:
            #     print("Robot is colliding with:")
            #     for contact in contact_points:
            #         bodyA = contact[1]
            #         bodyB = contact[2]
            #         nameA = p.getBodyInfo(bodyA)[1].decode("utf-8") or f"Unnamed (ID {bodyA})"
            #         nameB = p.getBodyInfo(bodyB)[1].decode("utf-8") or f"Unnamed (ID {bodyB})"
            #         contact_pos = contact[5]
            #         normal = contact[7]
            #         depth = contact[8]
            #         print(f" → {nameA} with {nameB}")
            #         print(f"    Contact pos: {contact_pos}, normal: {normal}, depth: {depth}")

            # Solve MPC
            # weights = np.array([w_goal, w_smooth, w_obs, 0.1])
            weights = np.array([10.0, 0.1, 0.01, 0.0])
            P = np.concatenate([state, path, obs_traj, static_flat, weights])
            lbx = [0.0, -1.0] * self.N + [-1e20] * 3 * (self.N + 1)
            ubx = [1.0, 1.0] * self.N + [1e20] * 3 * (self.N + 1)
            u0 = np.ones((2 * self.N + 3 * (self.N + 1), 1)) * 0.5
            g_len = self.N * 3 + 3


            # — assemble and reshape into a column vector —
            P = np.concatenate([state, path, obs_traj, static_flat, weights])
            P = P.reshape((-1, 1))   # now shape is (29,1)

            # — shape‐check instead of size‐check —
            expected_shape = self.solver.size_in('p')  # e.g. (29,1)
            print(f"[DEBUG] P.shape = {P.shape}, solver expects {expected_shape}")
            assert P.shape == expected_shape, (
                f"P-shape mismatch: got {P.shape}, solver wants {expected_shape}"
            )


            try:
                sol = self.solver(x0=u0, p=P, lbg=[0.0] * g_len, ubg=[0.0] * g_len, lbx=lbx, ubx=ubx)
                u_opt = sol['x'][:2 * self.N].full().reshape(self.N, 2)
            except Exception as e:
                print("Solver failed:", e)
                u_opt = np.zeros((self.N, 2))

            v, omega = u_opt[0]
            print(f"MPC Output: v={v}, omega={omega}")

            # Apply control to robot
            wheel_radius = 0.033
            wheel_base = 0.160
            v_left = (v - omega * wheel_base / 2) / wheel_radius
            v_right = (v + omega * wheel_base / 2) / wheel_radius

            if self.left_wheel_joint_id is not None and self.left_wheel_joint_id >= 0 \
            and self.right_wheel_joint_id is not None and self.right_wheel_joint_id >= 0:
                p.setJointMotorControl2(self.robot_id, self.left_wheel_joint_id,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=v_left,
                                        force=20.0)
                p.setJointMotorControl2(self.robot_id, self.right_wheel_joint_id,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity=v_right,
                                        force=20.0)

            # Update robot state
            self.theta = (self.theta + self.T * omega + np.pi) % (2 * np.pi) - np.pi
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            self.robot_pos = np.array(pos[:2])  # No scaling needed
            # Add debug line for robot (red)
            p.addUserDebugLine([self.robot_pos[0], self.robot_pos[1], 0], [self.robot_pos[0], self.robot_pos[1], 10], [1, 0, 0], lineWidth=3, lifeTime=0)

            # Update pedestrians
            for ped in self.peds:
                update_pedestrian(ped, self.peds, self.robot_pos)

            # Compute reward and done condition
            dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
            done = dist_to_goal < 0.2 or self.step_count >= self.max_steps
            reward = 1.0 if dist_to_goal < 0.2 else -0.01
            return self._get_observation(), reward, done, {}
        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            raise

    # def _get_observation(self):
    #     """Generate observation vector."""
    #     try:
    #         lidar = np.ones(16)  # Placeholder
    #         goal_vec = self.goal_pos - self.robot_pos
    #         goal_dist = np.linalg.norm(goal_vec)
    #         goal_dir = goal_vec / (goal_dist + 1e-6) if goal_dist > 0 else np.zeros(2)
    #         return np.concatenate([lidar, goal_dir, [goal_dist]]).astype(np.float32)
    #     except Exception as e:
    #         print(f"Error in _get_observation: {e}")
    #         traceback.print_exc()
    #         raise
    
    
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
        
        for ori, end in zip(origins, ends):
            p.addUserDebugLine(ori, end, [1,0,0], lifeTime=0.1)

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
        print("Environment initialized successfully")
        obs = env.reset(visualize=True)
        print("Reset successful")
        while True:
            action = np.array([1.0, 0.1, 0.1])
            obs, reward, done, _ = env.step(action)
            contacts = p.getContactPoints(bodyA=env.robot_id)
            if contacts:
                print(f"Robot is colliding with {len(contacts)} object(s):")
                for c in contacts:
                    print(f" → With body ID: {c[2]} (linkA: {c[3]}, linkB: {c[4]})")
                    print(f"   Contact position: {c[5]}, normal: {c[7]}, depth: {c[8]}")
            else:
                print("❌ No collision detected.")
            time.sleep(0.05)
            if done:
                print("Episode finished.")
                break
        print("Episode finished.")
    except Exception as e:
        print(f"Main loop error: {e}")
        traceback.print_exc()
        p.disconnect()  # Ensure PyBullet connection is closed