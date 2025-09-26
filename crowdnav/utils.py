import numpy as np
import traceback
import pybullet as p
import pybullet_data
import time
import gym
import matplotlib.pyplot as plt
from .planner import a_star, inflate_obstacles
import sys



MAX_STATIC=12
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
    """
    grid       : 2D numpy array (0=free, 1=inflated obstacle)
    start_idx  : tuple (row, col)
    goal_idx   : tuple (row, col)
    """
    plt.figure(figsize=(6,6))
    # show free=white, obstacle=black
    plt.imshow(grid, origin='lower', cmap='gray_r')  

    # unpack
    sr, sc = start_idx
    gr, gc = goal_idx

    # plot start as green star, goal as red X
    plt.scatter([sc], [sr], c='g', s=100, marker='*', label='Start')
    plt.scatter([gc], [gr], c='r', s=100, marker='X', label='Goal')

    plt.legend(loc='upper right')
    plt.title("Inflated Grid — Start ★, Goal ✕")
    plt.axis('off')
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

def update_astar_path(grid, start_w, goal_w, resolution):
    """
    Build a path from start_w -> goal_w on 'grid' (0=free, 1=occ).
    Returns world-space polyline as float array of shape (M,2).
    """
    import numpy as np
    from .planner import a_star, inflate_obstacles
    # Use the unified, battle-tested converters
    from .utils2.coord import world_to_grid as w2g
    from .utils2.coord import grid_to_world as g2w

    # --- sanitize inputs ---
    grid = np.asarray(grid, dtype=np.uint8)
    start_w = np.asarray(start_w, dtype=float).reshape(-1)
    goal_w  = np.asarray(goal_w,  dtype=float).reshape(-1)
    if start_w.size < 2 or goal_w.size < 2:
        raise ValueError(f"update_astar_path: start/goal must be 2D, got {start_w.shape}, {goal_w.shape}")

    # --- world -> grid indices (row, col) using the SAME convention everywhere ---
    start_idx = w2g(start_w[:2], float(resolution), grid.shape)
    goal_idx  = w2g(goal_w[:2],  float(resolution), grid.shape)
    

    # --- quick bounds/occ checks ---
    H, W = grid.shape
    def in_bounds(rc): return 0 <= rc[0] < H and 0 <= rc[1] < W
    if not in_bounds(start_idx) or not in_bounds(goal_idx):
        # no path
        return np.empty((0,2), dtype=float)
    if grid[start_idx] == 1 or grid[goal_idx] == 1:
        # fall back: don’t plan into obstacles
        return np.empty((0,2), dtype=float)

    # --- inflate obstacles (meters -> cells handled inside helper) ---
    ROBOT_RADIUS = 0.11
    INFLATION_SCALE = 0.8  # your choice above
    inflation_radius_m = ROBOT_RADIUS * INFLATION_SCALE
    inflated = inflate_obstacles(grid, inflation_radius_m, float(resolution))
    # ensure start/goal cells are free after inflation
    inflated[start_idx] = 0
    inflated[goal_idx]  = 0

    # --- run A* on inflated grid (returns list of (row,col)) ---
    path_rc = a_star(inflated, start_idx, goal_idx)
    if not path_rc:
        return np.empty((0,2), dtype=float)

    # normalize endpoints
    if path_rc[0] != start_idx:
        path_rc.insert(0, start_idx)
    if path_rc[-1] != goal_idx:
        path_rc.append(goal_idx)

    # --- convert grid path -> world polyline ---
    poly_world = np.empty((len(path_rc), 2), dtype=float)
    for k, rc in enumerate(path_rc):
        poly_world[k] = g2w(rc, float(resolution), grid.shape)  # (x,y) in meters

    return poly_world
