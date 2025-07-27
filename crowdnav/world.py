import pybullet as p
import numpy as np
import time
from .utils import MAP_SCALE



def create_occupancy_grid(resolution, map_size):
    """Generate an occupancy grid for the environment."""
    # Calculate grid dimensions
    L = map_size * MAP_SCALE
    # grid_size = int(map_size / resolution)
    grid_size = int(L / resolution)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)


    half_size = L / 2

    
    def world_to_grid(x, y):
       # x, y already scaled by MAP_SCALE when passed in
        col = int((x + half_size) / resolution)
        row = int((y + half_size) / resolution)
        return row, col

    # Mark wall and obstacle footprints (using unscaled coordinates)
    walls = []
    cylinders = []
    static_obs = []

    

    def mark_occupied(x_min, x_max, y_min, y_max):
        row_min, col_min = world_to_grid(x_min, y_min)
        row_max, col_max = world_to_grid(x_max, y_max)
        row_min = max(0, min(row_min, grid_size - 1))
        row_max = max(0, min(row_max, grid_size - 1))
        col_min = max(0, min(col_min, grid_size - 1))
        col_max = max(0, min(col_max, grid_size - 1))
        grid[row_min:row_max + 1, col_min:col_max + 1] = 1


    # Create walls and obstacles    
    # T, H = 25.0, 0.2, 1.0
    T = 0.2 * MAP_SCALE
    H = 1.0 * MAP_SCALE

    #Create object functions
    create_wall = lambda center, size: p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=size),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[center[0], center[1], center[2]]
    )
    
        # Cylinders
    create_cylinder = lambda center, radius, height: p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[center[0], center[1], center[2] + height / 2]
    )
    
    # Outer walls
    walls.append((create_wall([0, L/2, H/2], [L/2, T/2, H/2]), 0, L/2, H/2, L/2, T/2, H/2))  # Top
    walls.append((create_wall([0, -L/2, H/2], [L/2, T/2, H/2]), 0, -L/2, H/2, L/2, T/2, H/2))  # Bottom
    walls.append((create_wall([-L/2, 0, H/2], [T/2, L/2, H/2]), -L/2, 0, H/2, T/2, L/2, H/2))  # Left
    walls.append((create_wall([L/2, 0, H/2], [T/2, L/2, H/2]), L/2, 0, H/2, T/2, L/2, H/2))   # Right



    # === Inner walls (scaled by MAP_SCALE) ===
    walls.append((
        create_wall(
            [-8*MAP_SCALE,  10.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,  10.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,   7.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,   7.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,   4.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,   4.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,   1.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,   1.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,  -1.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,  -1.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,  -4.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,  -4.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE,  -7.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE,  -7.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [-8*MAP_SCALE, -10.5*MAP_SCALE, H/2],
            [(20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2]
        ),
        -8*MAP_SCALE, -10.5*MAP_SCALE, H/2,
        (20/6)*MAP_SCALE, 0.8*MAP_SCALE, H/2
    ))

    walls.append((
        create_wall(
            [-2*MAP_SCALE,   2*MAP_SCALE, H/2],
            [1*MAP_SCALE,   9*MAP_SCALE, H/2]
        ),
        -2*MAP_SCALE,   2*MAP_SCALE, H/2,
        1*MAP_SCALE,   9*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [ 1*MAP_SCALE,   2*MAP_SCALE, H/2],
            [1*MAP_SCALE,   9*MAP_SCALE, H/2]
        ),
        1*MAP_SCALE,    2*MAP_SCALE, H/2,
        1*MAP_SCALE,   9*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [ 4*MAP_SCALE,   2*MAP_SCALE, H/2],
            [1*MAP_SCALE,   9*MAP_SCALE, H/2]
        ),
        4*MAP_SCALE,    2*MAP_SCALE, H/2,
        1*MAP_SCALE,   9*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [ 7*MAP_SCALE,   2*MAP_SCALE, H/2],
            [1*MAP_SCALE,   9*MAP_SCALE, H/2]
        ),
        7*MAP_SCALE,    2*MAP_SCALE, H/2,
        1*MAP_SCALE,   9*MAP_SCALE, H/2
    ))
    walls.append((
        create_wall(
            [10*MAP_SCALE,   2*MAP_SCALE, H/2],
            [1*MAP_SCALE,   9*MAP_SCALE, H/2]
        ),
        10*MAP_SCALE,   2*MAP_SCALE, H/2,
        1*MAP_SCALE,   9*MAP_SCALE, H/2
    ))

    # === Cylinders (scaled) ===
    cylinders.append((
        create_cylinder(
            [ 10*MAP_SCALE, -10*MAP_SCALE, 0],
            1.5*MAP_SCALE, 1.0*MAP_SCALE
        ),
        10*MAP_SCALE, -10*MAP_SCALE, 0,
        1.5*MAP_SCALE, 1.0*MAP_SCALE
    ))
    cylinders.append((
        create_cylinder(
            [  4*MAP_SCALE, -10*MAP_SCALE, 0],
            1.5*MAP_SCALE, 1.0*MAP_SCALE
        ),
        4*MAP_SCALE,  -10*MAP_SCALE, 0,
        1.5*MAP_SCALE, 1.0*MAP_SCALE
    ))
    cylinders.append((
        create_cylinder(
            [ -2*MAP_SCALE, -10*MAP_SCALE, 0],
            1.5*MAP_SCALE, 1.0*MAP_SCALE
        ),
        -2*MAP_SCALE, -10*MAP_SCALE, 0,
        1.5*MAP_SCALE, 1.0*MAP_SCALE
    ))


    # Mark occupied areas on grid
    for wall_id, cx, cy, cz, sx, sy, sz in walls:
        x_min = cx - sx
        x_max = cx + sx
        y_min = cy - sy
        y_max = cy + sz
        mark_occupied(x_min, x_max, y_min, y_max)

    for cyl_id, cx, cy, cz, radius, height in cylinders:
        x_min = cx - radius
        x_max = cx + radius
        y_min = cy - radius
        y_max = cy + radius
        mark_occupied(x_min, x_max, y_min, y_max)



    half_L = L / 2
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:
                static_obs.append(np.array([
                    r * resolution + resolution/2 - half_L,
                    c * resolution + resolution/2 - half_L
                ]))


    return grid


def mark_box_on_grid(grid, center, size, resolution, map_size):
    half_size = (map_size * MAP_SCALE) / 2
    min_x = int(((center[0] - size[0]) + half_size) / resolution)
    max_x = int(((center[0] + size[0]) + half_size) / resolution)
    min_y = int(((center[1] - size[1]) + half_size) / resolution)
    max_y = int(((center[1] + size[1]) + half_size) / resolution)

    min_x = max(min_x, 0)
    max_x = min(max_x, grid.shape[0] - 1)
    min_y = max(min_y, 0)
    max_y = min(max_y, grid.shape[1] - 1)

    grid[min_x:max_x+1, min_y:max_y+1] = 1

def mark_cylinder_on_grid(grid, center, radius, resolution, map_size):
    half_size = (map_size * MAP_SCALE) / 2
    cx_idx = int((center[0] + half_size) / resolution)
    cy_idx = int((center[1] + half_size) / resolution)
    r_cells = int(np.ceil(radius / resolution))

    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            x_idx = cx_idx + dx
            y_idx = cy_idx + dy
            if (0 <= x_idx < grid.shape[0] and 0 <= y_idx < grid.shape[1]):
                dist = np.sqrt(dx**2 + dy**2) * resolution
                if dist <= radius:
                    grid[x_idx, y_idx] = 1

def create_world(client, resolution=0.1, map_size=25.0):
    p.resetSimulation(client)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    grid = create_occupancy_grid(resolution, map_size)
    L = map_size * MAP_SCALE
    half_L = L / 2

    robot_id = p.loadURDF(
        "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
        # basePosition=[1.0, 2.0, 0.1],
        basePosition=[1.0 * MAP_SCALE, 2.0 * MAP_SCALE, 0.1],
        useFixedBase=False
    )

    left_wheel_joint_id = None
    right_wheel_joint_id = None
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        if "wheel_left_joint" in joint_name:
            left_wheel_joint_id = i
        elif "wheel_right_joint" in joint_name:
            right_wheel_joint_id = i

    assert left_wheel_joint_id is not None, "Left wheel joint not found!"
    assert right_wheel_joint_id is not None, "Right wheel joint not found!"

    for _ in range(120):  # Simulate 2 seconds
        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    # L, T, H = map_size, 0.2, 1.0
    # Apply map scale to everything
    L = map_size * MAP_SCALE   # total world size
    T = 0.2 * MAP_SCALE        # wall thickness
    H = 1.0 * MAP_SCALE        # wall height

     
     
    create_wall = lambda center, size: p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=size),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=center
    )

    # === Outer Walls ===
    wall_defs = [
        ([0, half_L, H/2], [half_L, T/2, H/2]),
        ([0, -half_L, H/2], [half_L, T/2, H/2]),
        ([-half_L, 0, H/2], [T/2, half_L, H/2]),
        ([half_L, 0, H/2], [T/2, half_L, H/2])
    ]
    for center, size in wall_defs:
        create_wall(center, size)
        mark_box_on_grid(grid, center[:2], size[:2], resolution, map_size)

    static_obs = []  # Fill this if you need additional static obstacle metadata
    return robot_id, left_wheel_joint_id, right_wheel_joint_id, grid, static_obs