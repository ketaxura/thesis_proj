import numpy as np
import traceback
import pybullet as p
import pybullet_data
import time
import gym
import matplotlib.pyplot as plt
from .planner import a_star, inflate_obstacles
import sys



# Constants for the CrowdNavPyBulletEnv
MAP_SCALE = 0.45
WHEEL_RADIUS = 0.033  # TurtleBot3 wheel radius in meters
WHEEL_BASE = 0.160    # TurtleBot3 wheel base distance in meters
map_size=25.0

L = 25.0  # World length in meters
T = 0.07  # Wall thickness
H = 1.0   # Wall height
TURTLEBOT_RADIUS = 0.11  # Radius of TurtleBot3 Burger (half of 0.22m diameter)



def visualize_grid(grid, start_idx, goal_idx):
    vis_grid = np.copy(grid)
    vis_grid[start_idx[1], start_idx[0]] = 2  # Start
    vis_grid[goal_idx[1], goal_idx[0]] = 3    # Goal

    plt.imshow(vis_grid, cmap='gray')
    plt.title("Inflated Grid with Start (green) and Goal (red)")
    plt.colorbar()
    plt.show()



def get_random_position():
    """Generate a random position within the scaled world bounds, biased toward the interior."""
    half_size = 12.5  # Half of 25m map size
    # interior_margin = T + TURTLEBOT_RADIUS + 0.01  # Extra buffer
    interior_margin = 0
    min_x = -half_size + interior_margin
    max_x = half_size - interior_margin
    min_y = -half_size + interior_margin
    max_y = half_size - interior_margin
    return np.array([np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)])


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

        #  Add debugging print and visualization
        print("---- DEBUG A* INPUT ----")
        print("Grid shape:", grid.shape)
        print("Grid unique values:", np.unique(grid))
        print("Start index:", start_idx, "Value:", grid[start_idx])
        print("Goal index:", goal_idx, "Value:", grid[goal_idx])



        
        
        # Call A*
        # Save the original stdout
        original_stdout = sys.stdout
 


        with open("astar_debug_log.txt", "w") as f:
            sys.stdout = f  # Redirect prints to file

            try:
                
                # Example values — update based on your system
                robot_radius = TURTLEBOT_RADIUS           # meters
                grid_resolution = resolution       # meters per cell
                original_grid = grid

                
                inflated_grid = inflate_obstacles(original_grid, robot_radius, grid_resolution)
                visualize_grid(inflated_grid, start_idx, goal_idx)
                # plan on the inflated grid, not the raw one:
                path = a_star(inflated_grid, start_idx, goal_idx)
                
                
                
            except Exception as e:
                print("Exception during A*: ", e)
                path = []  # Ensure path is defined

            sys.stdout = original_stdout  # Reset to normal

        print(" A* debug log saved to astar_debug_log.txt")
                

        # Check the path only if it exists
        if not path:
            print(" Empty or failed A* path.")
            return []

        for (y, x) in path:
            if grid[y, x] == 1:
                print(f" A* path goes through obstacle at grid[{y}, {x}]")

        print(f" A* Path Length: {len(path)}")
        return path

    except Exception as e:
        print(f"Error in update_astar_path: {e}")
        import traceback
        traceback.print_exc()
        return []
