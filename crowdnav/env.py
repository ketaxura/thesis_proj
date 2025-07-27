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
from .utils import L, T, H, TURTLEBOT_RADIUS
from .utils import get_random_position, find_free_position, update_astar_path

import traceback


class CrowdNavPyBulletEnv(gym.Env):
    
    """Main Gym environment for CrowdNav with PyBullet."""
    def __init__(self, num_peds=1, max_static=1, max_steps=200, resolution=0.1):
        super().__init__()
        try:
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)

            # Environment parameters
            self.resolution = resolution
            self.max_steps = max_steps
            self.step_count = 0
            self.robot_pos = np.array([1.0, 1.0])  # Initial placeholder, will be randomized in reset
            self.theta = 0.0
            self.goal_pos = np.array([4.0, 4.0])  # Initial suggestion, will be validated
            self.num_peds = num_peds
            self.max_static = max_static
            self.peds = []
            self.static_obs = []
            self.half_size = L/2  # (or 12.5)

            # Initialize world and robot
            self.robot_id, self.left_wheel_joint_id, self.right_wheel_joint_id, self.grid, self.static_obs = create_world(
                self.client, resolution=self.resolution
            )
            
            self.grid_size = self.grid.shape
            # print(f"Grid Shape: {self.grid_size}")

            # MPC setup
            self.solver, self.f_dyn, self.T, self.N = build_mpc_solver_random_obs(max_obs=num_peds, max_static=max_static)

            # # Validate and set initial goal position
            # self.goal_pos = find_free_position(
            #     get_random_position(),
            #     "goal",
            #     self.grid,
            #     self.resolution,
            #     self.grid_size
            # )
            
            
            #We are getting start and goal poses inside of init, but why????
            goal_idx    = self.find_free_grid(label="goal")
            self.goal_pos = self.grid_to_world(goal_idx)
            
            # Observation and action spaces
            obs_dim = 16 + 2 + 1  # LIDAR + goal_dir + goal_dist
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
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
        """(row,col) → (x,y) [m]."""
        x = idx[1] * self.resolution - self.half_size
        y = idx[0] * self.resolution - self.half_size
        return np.array([x, y])
    
    
    def find_free_grid(self, label, avoid=None, min_dist=0.5, max_attempts=200):
        """Sample random grid‐cells until we find one that’s free and far enough from `avoid`."""
        for _ in range(max_attempts):
            pos = get_random_position()
            idx = self.world_to_grid(pos)
            if self.grid[idx] == 1:
                world = self.grid_to_world(idx)
                if avoid is None or np.linalg.norm(world - avoid) >= min_dist:
                    return idx
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
        """Reset the environment to initial state with random start position."""
        try:
            self.step_count = 0
            max_attempts = 100

            # start_idx = self.find_free_grid(label="start", avoid=self.goal_pos)
            start_idx=[126, 61]
            self.robot_pos = self.grid_to_world(start_idx)

            self.theta = 0.0
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [self.robot_pos[0], self.robot_pos[1], 0.1],
                p.getQuaternionFromEuler([0, 0, self.theta])
            )
            goal_idx =[33,25]
            # goal_idx    = self.find_free_grid(label="goal", avoid=self.robot_pos)
            self.goal_pos = self.grid_to_world(goal_idx)    

        # Robot pos: [-6.4  0.1], Grid idx: [ 61 126]
        # Goal pos: [-10.6 -10.7], Grid idx: [19 18]




            # inside reset(), after picking start & goal:
            self.start_idx = self.world_to_grid(self.robot_pos)
            self.goal_idx  = self.world_to_grid(self.goal_pos)

            # Spawn pedestrians
            self.peds = [create_random_pedestrian(self.robot_pos, self.client) for _ in range(self.num_peds)]

            print(f"Robot pos: {self.robot_pos}, Grid idx: {((self.robot_pos + 12.5) / self.resolution).astype(int)}")
            print(f"Goal pos: {self.goal_pos}, Grid idx: {((self.goal_pos + 12.5) / self.resolution).astype(int)}")

            # Plan new A* path
            self.global_path_idx = update_astar_path(
                self.robot_pos,
                self.goal_pos,
                self.grid,
                self.resolution
            )


            self.show_occupancy_grid(self.grid, self.global_path_idx)

            # Visual goal line
            p.addUserDebugLine(
                [self.goal_pos[0], self.goal_pos[1], 0],
                [self.goal_pos[0], self.goal_pos[1], 10],
                [0, 1, 0], lineWidth=3, lifeTime=0
            )

            # # Optional debug info
            # print("\nScene Object Summary:")
            # for i in range(p.getNumBodies()):
            #     name = p.getBodyInfo(i)[1].decode("utf-8") or f"Unnamed (ID {i})"
            #     print(f"Body ID {i}: {name}")
            # print("Floor collision info:")
            # for i in range(p.getNumBodies()):
            #     name = p.getBodyInfo(i)[1].decode("utf-8") or f"Unnamed (ID {i})"
            #     if 'plane' in name.lower() or 'ground' in name.lower():
            #         print(p.getCollisionShapeData(i, -1))
            #         pos, orn = p.getBasePositionAndOrientation(i)
            #         print(f"{name} origin: {pos}")

            # # Visualize occupancy grid + path (only when requested)
            # if visualize:
            #     self._plot_environment_debug()



            return self._get_observation()


        except Exception as e:
            print(f"Error in reset: {e}")
            traceback.print_exc()
            raise





    def _plot_astar_path(self):
        """Plot the A* path on the occupancy grid."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid.T, origin='lower', cmap='gray')

        # Plot A* path if available
        if self.global_path_idx is not None and len(self.global_path_idx) > 0:
            path = np.array(self.global_path_idx)
            plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=2, label='A* Path')  # path is in (row, col)
            plt.plot(path[0, 1], path[0, 0], 'go', label='Start')  # Start
            plt.plot(path[-1, 1], path[-1, 0], 'ro', label='Goal')  # Goal
        else:
            print("Warning: A* path is empty or not initialized.")

        plt.legend()
        plt.title("A* Path on Occupancy Grid")
        plt.tight_layout()
        plt.grid(False)
        plt.show()
        plt.close()


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
                half_size = 12.5
                sampled = [((self.goal_pos + half_size) / self.resolution).astype(int)] * self.N

            elif len(self.global_path_idx) < self.N:
                sampled = self.global_path_idx + [self.global_path_idx[-1]] * (self.N - len(self.global_path_idx))
            else:
                sampled = np.linspace(0, len(self.global_path_idx) - 1, self.N).astype(int)
                sampled = [self.global_path_idx[i] for i in sampled]
            # Convert grid indices to unscaled world coordinates
            half_size = 12.5
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
            weights = np.array([1.0, 0.1, 0.01, 0.1])
            P = np.concatenate([state, path, obs_traj, static_flat, weights])
            lbx = [0.0, -1.0] * self.N + [-1e20] * 3 * (self.N + 1)
            ubx = [1.0, 1.0] * self.N + [1e20] * 3 * (self.N + 1)
            u0 = np.ones((2 * self.N + 3 * (self.N + 1), 1)) * 0.1
            g_len = self.N * 3 + 3

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

    def _get_observation(self):
        """Generate observation vector."""
        try:
            lidar = np.ones(16)  # Placeholder
            goal_vec = self.goal_pos - self.robot_pos
            goal_dist = np.linalg.norm(goal_vec)
            goal_dir = goal_vec / (goal_dist + 1e-6) if goal_dist > 0 else np.zeros(2)
            return np.concatenate([lidar, goal_dir, [goal_dist]]).astype(np.float32)
        except Exception as e:
            print(f"Error in _get_observation: {e}")
            traceback.print_exc()
            raise



    def show_occupancy_grid(self, grid, path=None):
        plt.figure(figsize=(6,6))
        plt.imshow(grid.T, origin='lower', cmap='gray_r')

        # Always plot start & goal
        sr, sc = self.start_idx
        gr, gc = self.goal_idx
        plt.scatter([sc], [sr], c='g', s=100, marker='*', label='Start')
        plt.scatter([gc], [gr], c='r', s=100, marker='X', label='Goal')

        # If A* path exists, plot it too
        if path:
            arr = np.array(path)
            rows, cols = arr[:,0], arr[:,1]
            plt.plot(rows, cols, 'b-', linewidth=2, label='A* Path')
        else:
            plt.text(0.5, 0.95, "No valid A* path", color='orange',
                    transform=plt.gca().transAxes,
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        plt.legend(loc='upper right')
        plt.title("Occupancy Grid — Start (★), Goal (✕)")
        plt.grid(False)
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