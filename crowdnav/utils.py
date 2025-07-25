import numpy as np
import traceback
import pybullet as p
import pybullet_data
import time
import gym
import matplotlib.pyplot as plt
from .planner import a_star




# Constants for the CrowdNavPyBulletEnv
MAP_SCALE = 0.35
WHEEL_RADIUS = 0.033  # TurtleBot3 wheel radius in meters
WHEEL_BASE = 0.160    # TurtleBot3 wheel base distance in meters


L = 25.0  # World length in meters
T = 0.07  # Wall thickness
H = 1.0   # Wall height
TURTLEBOT_RADIUS = 0.11  # Radius of TurtleBot3 Burger (half of 0.22m diameter)


def get_random_position():
    """Generate a random position within the scaled world bounds, biased toward the interior."""
    half_size = 12.5  # Half of 25m map size
    interior_margin = T + TURTLEBOT_RADIUS + 0.01  # Extra buffer
    min_x = -half_size + interior_margin
    max_x = half_size - interior_margin
    min_y = -half_size + interior_margin
    max_y = half_size - interior_margin
    return np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])



# def find_free_position(pos, label, grid, resolution, grid_size, robot_radius=0.11, wall_thickness=0.07, min_distance_from=None):
#     """Find the nearest free position in the grid for a given position, accounting for robot radius and map interior."""
#     half_size = 12.5  # Half of 25m map size
#     # Define an interior region inside the walls (excluding 0.07m wall thickness)
#     interior_margin = 0.07  # Wall thickness
#     min_x = -half_size + interior_margin
#     max_x = half_size - interior_margin
#     min_y = -half_size + interior_margin
#     max_y = half_size - interior_margin
#     if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
#         pos = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
#         print(f"Adjusting {label} position {pos} to be inside the map")

#     x_idx = int((pos[0] + half_size) / self.resolution)
#     y_idx = int((pos[1] + half_size) / self.resolution)
#     # Check a neighborhood around the position based on robot radius
#     radius_cells = int(np.ceil(TURTLEBOT_RADIUS / self.resolution))
#     is_free = True
#     for dx in range(-radius_cells, radius_cells + 1):
#         for dy in range(-radius_cells, radius_cells + 1):
#             new_x_idx = x_idx + dx
#             new_y_idx = y_idx + dy
#             if (0 <= new_x_idx < self.grid_size[0] and 0 <= new_y_idx < self.grid_size[1] and
#                     self.grid[new_x_idx, new_y_idx] == 1):
#                 is_free = False
#                 break
#         if not is_free:
#             break
#     if (0 <= x_idx < self.grid_size[0] and 0 <= y_idx < self.grid_size[1] and is_free and
#             (min_distance_from is None or np.linalg.norm(pos - min_distance_from) >= 0.5)):
#         print(f"{label} position {pos} is free at grid index ({x_idx}, {y_idx})")
#         return pos
#     print(f"Warning: {label} position {pos} at grid index ({x_idx}, {y_idx}) is invalid, occupied, or too close")
#     # Enhanced search for a free position
#     for r in range(1, max(self.grid_size) * 2):  # Increased search radius
#         for dx in range(-r, r + 1):
#             for dy in range(-r, r + 1):
#                 new_x_idx = x_idx + dx
#                 new_y_idx = y_idx + dy
#                 if 0 <= new_x_idx < self.grid_size[0] and 0 <= new_y_idx < self.grid_size[1]:
#                     is_free = True
#                     for dx2 in range(-radius_cells, radius_cells + 1):
#                         for dy2 in range(-radius_cells, radius_cells + 1):
#                             check_x = new_x_idx + dx2
#                             check_y = new_y_idx + dy2
#                             if (0 <= check_x < self.grid_size[0] and 0 <= check_y < self.grid_size[1] and
#                                     self.grid[check_x, check_y] == 1):
#                                 is_free = False
#                                 break
#                         if not is_free:
#                             break
#                     if is_free and (min_distance_from is None or np.linalg.norm(
#                             np.array([new_x_idx * self.resolution - half_size, new_y_idx * self.resolution - half_size]) - min_distance_from) >= 0.5):
#                         new_pos = np.array([
#                             new_x_idx * self.resolution - half_size,
#                             new_y_idx * self.resolution - half_size
#                         ])
#                         if min_x <= new_pos[0] <= max_x and min_y <= new_pos[1] <= max_y:
#                             print(f"Found free {label} position {new_pos} at grid index ({new_x_idx}, {new_y_idx})")
#                             return new_pos
#     raise ValueError(f"Could not find a free {label} position inside the map near {pos} with robot radius buffer after extensive search")


def find_free_position(pos, label, grid, resolution, grid_size, robot_radius=0.11, wall_thickness=0.07, min_distance_from=None):
    half_size = 12.5
    min_x = -half_size + wall_thickness
    max_x = half_size - wall_thickness
    min_y = -half_size + wall_thickness
    max_y = half_size - wall_thickness

    if not (min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y):
        pos = np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])
        print(f"Adjusting {label} position {pos} to be inside the map")

    x_idx = int((pos[0] + half_size) / resolution)
    y_idx = int((pos[1] + half_size) / resolution)
    radius_cells = int(np.ceil(robot_radius / resolution))

    def is_cell_free(cx, cy):
        return (
            0 <= cx < grid_size[0] and 0 <= cy < grid_size[1] and grid[cx, cy] == 0
        )

    is_free = all(
        is_cell_free(x_idx + dx, y_idx + dy)
        for dx in range(-radius_cells, radius_cells + 1)
        for dy in range(-radius_cells, radius_cells + 1)
    )

    if is_free and (
        min_distance_from is None or np.linalg.norm(pos - min_distance_from) >= 0.5
    ):
        print(f"{label} position {pos} is free at grid index ({x_idx}, {y_idx})")
        return pos

    print(f"Warning: {label} position {pos} at grid index ({x_idx}, {y_idx}) is invalid, occupied, or too close")

    for r in range(1, max(grid_size) * 2):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                new_x = x_idx + dx
                new_y = y_idx + dy
                if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                    new_pos = np.array([
                        new_x * resolution - half_size,
                        new_y * resolution - half_size
                    ])
                    is_free = all(
                        is_cell_free(new_x + dx2, new_y + dy2)
                        for dx2 in range(-radius_cells, radius_cells + 1)
                        for dy2 in range(-radius_cells, radius_cells + 1)
                    )
                    if is_free and (
                        min_distance_from is None or np.linalg.norm(new_pos - min_distance_from) >= 0.5
                    ):
                        if min_x <= new_pos[0] <= max_x and min_y <= new_pos[1] <= max_y:
                            print(f"Found free {label} position {new_pos} at grid index ({new_x}, {new_y})")
                            return new_pos

    raise ValueError(f"Could not find a free {label} position inside the map near {pos}")




def update_astar_path(robot_pos, goal_pos, grid, resolution):
    import matplotlib.pyplot as plt

    try:
        half_size = 12.5
        start_idx = tuple(((robot_pos + half_size) / resolution).astype(int))
        goal_idx = tuple(((goal_pos + half_size) / resolution).astype(int))
        grid_size = grid.shape

        if not (0 <= start_idx[0] < grid_size[0] and 0 <= start_idx[1] < grid_size[1]):
            print(f"Start index {start_idx} out of bounds.")
            return []

        if not (0 <= goal_idx[0] < grid_size[0] and 0 <= goal_idx[1] < grid_size[1]):
            print(f"Goal index {goal_idx} out of bounds.")
            return []

        if grid[start_idx[0], start_idx[1]] == 1:
            print("Start position is inside obstacle.")
            return []

        if grid[goal_idx[0], goal_idx[1]] == 1:
            print("Goal position is inside obstacle.")
            return []

        # 💡 Add debugging print and visualization
        print("---- DEBUG A* INPUT ----")
        print("Grid shape:", grid.shape)
        print("Grid unique values:", np.unique(grid))
        print("Start index:", start_idx, "Value:", grid[start_idx])
        print("Goal index:", goal_idx, "Value:", grid[goal_idx])

        debug_grid = grid.copy()
        debug_grid[start_idx] = 0.5  # Mark start
        debug_grid[goal_idx] = 0.8   # Mark goal
        plt.imshow(1 - debug_grid.T, origin='lower', cmap='gray')
        plt.title("Debug Grid with Start and Goal")
        plt.colorbar()
        plt.show()

        # 🔁 Call A*
        path = a_star(grid, start_idx, goal_idx)
        for (y, x) in path:
            if grid[y, x] == 1:
                print(f"❌ A* path goes through obstacle at grid[{y}, {x}]")

        print(f"A* Path Length: {len(path)}")
        return path

    except Exception as e:
        print(f"Error in update_astar_path: {e}")
        import traceback
        traceback.print_exc()
        return []
