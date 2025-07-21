import pybullet as p
import pybullet_data
import numpy as np
import time
import gym

from gym import spaces
from .planner import a_star
from .mpc import build_mpc_solver_random_obs
from .world import create_world
from .pedestrian import create_random_pedestrian, update_pedestrian

import traceback
#added comments

TURTLEBOT_RADIUS = 0.11  # Radius of TurtleBot3 Burger (half of 0.22m diameter)

class CrowdNavPyBulletEnv(gym.Env):
    """Main Gym environment for CrowdNav with PyBullet."""
    def __init__(self, num_peds=2, max_static=1, max_steps=200, resolution=0.1):
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

            # Initialize world and robot
            self.robot_id, self.left_wheel_joint_id, self.right_wheel_joint_id, self.grid, self.static_obs = create_world(
                self.client, resolution=self.resolution
            )
            self.grid_size = self.grid.shape
            print(f"Grid Shape: {self.grid_size}")

            # MPC setup
            self.solver, self.f_dyn, self.T, self.N = build_mpc_solver_random_obs(max_obs=num_peds, max_static=max_static)

            # Validate and set initial goal position
            self.goal_pos = self._find_free_position(self._get_random_position(), "goal")

            # Observation and action spaces
            obs_dim = 16 + 2 + 1  # LIDAR + goal_dir + goal_dist
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

            # Visualize grid for debugging
            import matplotlib.pyplot as plt
            plt.imshow(self.grid.T, origin='lower', cmap='binary')
            plt.plot(self.robot_pos[1]/self.resolution, self.robot_pos[0]/self.resolution, 'ro', label='Robot')
            plt.plot(self.goal_pos[1]/self.resolution, self.goal_pos[0]/self.resolution, 'go', label='Goal')
            plt.legend()
            plt.title("Occupancy Grid")
            plt.show()

        except Exception as e:
            print(f"Error in __init__: {e}")
            traceback.print_exc()
            raise

    def _find_free_position(self, pos, label, min_distance_from=None):
        """Find the nearest free position in the grid for a given position, accounting for robot radius and map interior."""
        half_size = 12.5  # Half of 25m map size
        # Define an interior region inside the walls (excluding 0.07m wall thickness)
        interior_margin = 0.07  # Wall thickness
        min_x = -half_size + interior_margin
        max_x = half_size - interior_margin
        min_y = -half_size + interior_margin
        max_y = half_size - interior_margin
        if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
            pos = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
            print(f"Adjusting {label} position {pos} to be inside the map")

        x_idx = int((pos[0] + half_size) / self.resolution)
        y_idx = int((pos[1] + half_size) / self.resolution)
        # Check a neighborhood around the position based on robot radius
        radius_cells = int(np.ceil(TURTLEBOT_RADIUS / self.resolution))
        is_free = True
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                new_x_idx = x_idx + dx
                new_y_idx = y_idx + dy
                if (0 <= new_x_idx < self.grid_size[0] and 0 <= new_y_idx < self.grid_size[1] and
                        self.grid[new_x_idx, new_y_idx] == 1):
                    is_free = False
                    break
            if not is_free:
                break
        if (0 <= x_idx < self.grid_size[0] and 0 <= y_idx < self.grid_size[1] and is_free and
                (min_distance_from is None or np.linalg.norm(pos - min_distance_from) >= 0.5)):
            print(f"{label} position {pos} is free at grid index ({x_idx}, {y_idx})")
            return pos
        print(f"Warning: {label} position {pos} at grid index ({x_idx}, {y_idx}) is invalid, occupied, or too close")
        # Enhanced search for a free position
        for r in range(1, max(self.grid_size) * 2):  # Increased search radius
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    new_x_idx = x_idx + dx
                    new_y_idx = y_idx + dy
                    if 0 <= new_x_idx < self.grid_size[0] and 0 <= new_y_idx < self.grid_size[1]:
                        is_free = True
                        for dx2 in range(-radius_cells, radius_cells + 1):
                            for dy2 in range(-radius_cells, radius_cells + 1):
                                check_x = new_x_idx + dx2
                                check_y = new_y_idx + dy2
                                if (0 <= check_x < self.grid_size[0] and 0 <= check_y < self.grid_size[1] and
                                        self.grid[check_x, check_y] == 1):
                                    is_free = False
                                    break
                            if not is_free:
                                break
                        if is_free and (min_distance_from is None or np.linalg.norm(
                                np.array([new_x_idx * self.resolution - half_size, new_y_idx * self.resolution - half_size]) - min_distance_from) >= 0.5):
                            new_pos = np.array([
                                new_x_idx * self.resolution - half_size,
                                new_y_idx * self.resolution - half_size
                            ])
                            if min_x <= new_pos[0] <= max_x and min_y <= new_pos[1] <= max_y:
                                print(f"Found free {label} position {new_pos} at grid index ({new_x_idx}, {new_y_idx})")
                                return new_pos
        raise ValueError(f"Could not find a free {label} position inside the map near {pos} with robot radius buffer after extensive search")

    def _get_random_position(self):
        """Generate a random position within the scaled world bounds, biased toward the interior."""
        half_size = 12.5  # Half of 25m map size
        interior_margin = 0.07  # Wall thickness
        min_x = -half_size + interior_margin
        max_x = half_size - interior_margin
        min_y = -half_size + interior_margin
        max_y = half_size - interior_margin
        return np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])

    def reset(self):
        """Reset the environment to initial state with random start position."""
        try:
            self.step_count = 0
            # Generate random start position until a valid one is found
            max_attempts = 100
            for _ in range(max_attempts):
                candidate_pos = self._get_random_position()
                try:
                    self.robot_pos = self._find_free_position(candidate_pos, "start", min_distance_from=self.goal_pos)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Could not find a valid random start position after maximum attempts")
            self.theta = 0.0
            # Position for PyBullet (no scaling needed)
            p.resetBasePositionAndOrientation(self.robot_id, [self.robot_pos[0], self.robot_pos[1], 0.1],
                                              p.getQuaternionFromEuler([0, 0, self.theta]))
            # Re-validate goal position to ensure it remains free
            self.goal_pos = self._find_free_position(self._get_random_position(), "goal", min_distance_from=self.robot_pos)
            self.peds = [create_random_pedestrian(self.robot_pos, self.client) for _ in range(self.num_peds)]
            self._update_astar_path()

            # Add debug line for goal (green)
            p.addUserDebugLine([self.goal_pos[0], self.goal_pos[1], 0], [self.goal_pos[0], self.goal_pos[1], 10], [0, 1, 0], lineWidth=3, lifeTime=0)

            # Debug scene objects
            print("\nScene Object Summary:")
            for i in range(p.getNumBodies()):
                name = p.getBodyInfo(i)[1].decode("utf-8") or f"Unnamed (ID {i})"
                print(f"Body ID {i}: {name}")
            print("Floor collision info:")
            for i in range(p.getNumBodies()):
                name = p.getBodyInfo(i)[1].decode("utf-8") or f"Unnamed (ID {i})"
                if 'plane' in name.lower() or 'ground' in name.lower():
                    print(p.getCollisionShapeData(i, -1))
                    pos, orn = p.getBasePositionAndOrientation(i)
                    print(f"{name} origin: {pos}")

            # Update grid visualization
            import matplotlib.pyplot as plt
            plt.imshow(self.grid.T, origin='lower', cmap='binary')
            plt.plot(self.robot_pos[1]/self.resolution, self.robot_pos[0]/self.resolution, 'ro', label='Robot')
            plt.plot(self.goal_pos[1]/self.resolution, self.goal_pos[0]/self.resolution, 'go', label='Goal')
            plt.legend()
            plt.title("Occupancy Grid")
            plt.show()

            return self._get_observation()
        except Exception as e:
            print(f"Error in reset: {e}")
            traceback.print_exc()
            raise

    def _update_astar_path(self):
        """Update the global A* path from robot to goal."""
        try:
            # Convert unscaled positions to grid indices
            half_size = 12.5  # Half of 25m map size
            start_idx = tuple(((self.robot_pos + half_size) / self.resolution).astype(int))
            goal_idx = tuple(((self.goal_pos + half_size) / self.resolution).astype(int))
            print(f"Start Index: {start_idx}, Goal Index: {goal_idx}, Grid Value at Goal: {self.grid[goal_idx]}")
            # Check if indices are within grid bounds
            if not (0 <= start_idx[0] < self.grid_size[0] and 0 <= start_idx[1] < self.grid_size[1]):
                print(f"Warning: Start index {start_idx} out of grid bounds {self.grid_size}")
                self.global_path_idx = []
                return
            if not (0 <= goal_idx[0] < self.grid_size[0] and 0 <= goal_idx[1] < self.grid_size[1]):
                print(f"Warning: Goal index {goal_idx} out of grid bounds {self.grid_size}")
                self.global_path_idx = []
                return
            # Check if start or goal is in an obstacle
            if self.grid[start_idx[0], start_idx[1]] == 1:
                print(f"Warning: Start position {self.robot_pos} is in an obstacle")
                self.global_path_idx = []
                return
            if self.grid[goal_idx[0], goal_idx[1]] == 1:
                print(f"Warning: Goal position {self.goal_pos} is in an obstacle at index {goal_idx}")
                self.global_path_idx = []
                return
            self.global_path_idx = a_star(self.grid, start_idx, goal_idx)
            print(f"A* Path Length: {len(self.global_path_idx)}")
            if not self.global_path_idx:
                print("Warning: A* returned an empty path")
        except Exception as e:
            print(f"Error in _update_astar_path: {e}")
            traceback.print_exc()
            self.global_path_idx = []
            raise

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
                sampled = [(self.goal_pos + half_size) / self.resolution.astype(int)] * self.N
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
            if contact_points:
                print("Robot is colliding with:")
                for contact in contact_points:
                    bodyA = contact[1]
                    bodyB = contact[2]
                    nameA = p.getBodyInfo(bodyA)[1].decode("utf-8") or f"Unnamed (ID {bodyA})"
                    nameB = p.getBodyInfo(bodyB)[1].decode("utf-8") or f"Unnamed (ID {bodyB})"
                    contact_pos = contact[5]
                    normal = contact[7]
                    depth = contact[8]
                    print(f" → {nameA} with {nameB}")
                    print(f"    Contact pos: {contact_pos}, normal: {normal}, depth: {depth}")

            # Solve MPC
            weights = np.array([w_goal, w_smooth, w_obs, 0.1])
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

if __name__ == "__main__":
    try:
        env = CrowdNavPyBulletEnv()
        print("Environment initialized successfully")
        obs = env.reset()
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