import pybullet as p
import numpy as np
import time

def create_occupancy_grid(resolution, map_size):
    """Generate an occupancy grid for the environment."""
    # Calculate grid dimensions
    grid_size = int(map_size / resolution)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Convert world coordinates to grid indices
    def world_to_grid(x, y, resolution, map_size):
        half_size = map_size / 2
        x_idx = int((x + half_size) / resolution)
        y_idx = int((y + half_size) / resolution)
        return x_idx, y_idx

    # Mark wall and obstacle footprints (using unscaled coordinates)
    walls = []
    cylinders = []
    static_obs = []

    def mark_occupied(x_min, x_max, y_min, y_max):
        x_min_idx, y_min_idx = world_to_grid(x_min, y_min, resolution, map_size)
        x_max_idx, y_max_idx = world_to_grid(x_max, y_max, resolution, map_size)
        x_min_idx = max(0, min(x_min_idx, grid_size - 1))
        x_max_idx = max(0, min(x_max_idx, grid_size - 1))
        y_min_idx = max(0, min(y_min_idx, grid_size - 1))
        y_max_idx = max(0, min(y_max_idx, grid_size - 1))
        grid[x_min_idx:x_max_idx + 1, y_min_idx:y_max_idx + 1] = 1

    # Create walls and obstacles
    L, T, H = 25.0, 0.2, 1.0

    # Outer walls
    create_wall = lambda center, size: p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=size),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[center[0], center[1], center[2]]
    )
    walls.append((create_wall([0, L/2, H/2], [L/2, T/2, H/2]), 0, L/2, H/2, L/2, T/2, H/2))  # Top
    walls.append((create_wall([0, -L/2, H/2], [L/2, T/2, H/2]), 0, -L/2, H/2, L/2, T/2, H/2))  # Bottom
    walls.append((create_wall([-L/2, 0, H/2], [T/2, L/2, H/2]), -L/2, 0, H/2, T/2, L/2, H/2))  # Left
    walls.append((create_wall([L/2, 0, H/2], [T/2, L/2, H/2]), L/2, 0, H/2, T/2, L/2, H/2))   # Right

    # Inner walls
    walls.append((create_wall([-8, 10.5, H/2], [20/6, 0.8, H/2]), -8, 10.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, 7.5, H/2], [20/6, 0.8, H/2]), -8, 7.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, 4.5, H/2], [20/6, 0.8, H/2]), -8, 4.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, 1.5, H/2], [20/6, 0.8, H/2]), -8, 1.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, -1.5, H/2], [20/6, 0.8, H/2]), -8, -1.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, -4.5, H/2], [20/6, 0.8, H/2]), -8, -4.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, -7.5, H/2], [20/6, 0.8, H/2]), -8, -7.5, H/2, 20/6, 0.8, H/2))
    walls.append((create_wall([-8, -10.5, H/2], [20/6, 0.8, H/2]), -8, -10.5, H/2, 20/6, 0.8, H/2))

    walls.append((create_wall([-2, 2, H/2], [1, 9, H/2]), -2, 2, H/2, 1, 9, H/2))
    walls.append((create_wall([1, 2, H/2], [1, 9, H/2]), 1, 2, H/2, 1, 9, H/2))
    walls.append((create_wall([4, 2, H/2], [1, 9, H/2]), 4, 2, H/2, 1, 9, H/2))
    walls.append((create_wall([7, 2, H/2], [1, 9, H/2]), 7, 2, H/2, 1, 9, H/2))
    walls.append((create_wall([10, 2, H/2], [1, 9, H/2]), 10, 2, H/2, 1, 9, H/2))

    # Cylinders
    create_cylinder = lambda center, radius, height: p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.5, 0.5, 0.5, 1]),
        basePosition=[center[0], center[1], center[2] + height / 2]
    )
    cylinders.append((create_cylinder([10, -10, 0], 1.5, 1.0), 10, -10, 0, 1.5, 1.0))
    cylinders.append((create_cylinder([4, -10, 0], 1.5, 1.0), 4, -10, 0, 1.5, 1.0))
    cylinders.append((create_cylinder([-2, -10, 0], 1.5, 1.0), -2, -10, 0, 1.5, 1.0))

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

    # Populate static_obs from grid
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                static_obs.append(np.array([
                    x * resolution + resolution / 2 - (map_size / 2),
                    y * resolution + resolution / 2 - (map_size / 2)
                ]))

    return grid


def mark_box_on_grid(grid, center, size, resolution, map_size):
    half_size = map_size / 2
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
    half_size = map_size / 2
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

    robot_id = p.loadURDF(
        "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
        basePosition=[1.0, 2.0, 0.1],
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

    L, T, H = map_size, 0.2, 1.0
    half_L = L / 2

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

    # === Inner Walls ===
    inner_wall_defs = [
        ([-8, 10.5, H/2], [20/6, 0.8, H/2]),
        ([-8, 7.5, H/2], [20/6, 0.8, H/2]),
        ([-8, 4.5, H/2], [20/6, 0.8, H/2]),
        ([-8, 1.5, H/2], [20/6, 0.8, H/2]),
        ([-8, -1.5, H/2], [20/6, 0.8, H/2]),
        ([-8, -4.5, H/2], [20/6, 0.8, H/2]),
        ([-8, -7.5, H/2], [20/6, 0.8, H/2]),
        ([-8, -10.5, H/2], [20/6, 0.8, H/2]),
        ([-2, 2, H/2], [1, 9, H/2]),
        ([1, 2, H/2], [1, 9, H/2]),
        ([4, 2, H/2], [1, 9, H/2]),
        ([7, 2, H/2], [1, 9, H/2]),
        ([10, 2, H/2], [1, 9, H/2])
    ]
    for center, size in inner_wall_defs:
        create_wall(center, size)
        mark_box_on_grid(grid, center[:2], size[:2], resolution, map_size)

    # === Cylinders ===
    cylinder_defs = [
        ([10, -10, 0], 1.5, 1.0),
        ([4, -10, 0], 1.5, 1.0),
        ([-2, -10, 0], 1.5, 1.0)
    ]
    for center, radius, height in cylinder_defs:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.5, 0.5, 0.5, 1]),
            basePosition=[center[0], center[1], center[2] + height / 2]
        )
        mark_cylinder_on_grid(grid, center[:2], radius, resolution, map_size)

    static_obs = []  # Fill this if you need additional static obstacle metadata
    return robot_id, left_wheel_joint_id, right_wheel_joint_id, grid, static_obs