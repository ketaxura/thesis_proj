import pybullet as p
import numpy as np
import time
from .utils import MAP_SCALE



# def create_occupancy_grid(resolution, map_size):
#     """Generate an occupancy grid for the environment."""
#     # Calculate grid dimensions
#     L = map_size * MAP_SCALE
#     # grid_size = int(map_size / resolution)
#     grid_size = int(L / resolution)
#     grid = np.zeros((grid_size, grid_size), dtype=np.uint8)


#     half_size = L / 2

    
#     def world_to_grid(x, y):
#        # x, y already scaled by MAP_SCALE when passed in
#         col = int((x + half_size) / resolution)
#         row = int((y + half_size) / resolution)
#         return row, col

#     # Mark wall and obstacle footprints (using unscaled coordinates)
#     walls = []
#     cylinders = []
#     static_obs = []

    

#     def mark_occupied(x_min, x_max, y_min, y_max):
#         row_min, col_min = world_to_grid(x_min, y_min)
#         row_max, col_max = world_to_grid(x_max, y_max)
#         row_min = max(0, min(row_min, grid_size - 1))
#         row_max = max(0, min(row_max, grid_size - 1))
#         col_min = max(0, min(col_min, grid_size - 1))
#         col_max = max(0, min(col_max, grid_size - 1))
#         grid[row_min:row_max + 1, col_min:col_max + 1] = 1


#     # Create walls and obstacles    
#     # T, H = 25.0, 0.2, 1.0
#     T = 0.2 * MAP_SCALE
#     H = 1.0 * MAP_SCALE

#     #Create object functions
#     create_wall = lambda center, size: p.createMultiBody(
#         baseMass=0,
#         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=size),
#         baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.5, 0.5, 0.5, 1]),
#         basePosition=[center[0], center[1], center[2]]
#     )
    
#         # Cylinders
#     create_cylinder = lambda center, radius, height: p.createMultiBody(
#         baseMass=0,
#         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height),
#         baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=[0.5, 0.5, 0.5, 1]),
#         basePosition=[center[0], center[1], center[2] + height / 2]
#     )
    
#     # Outer walls
#     walls.append((create_wall([0, L/2, H/2], [L/2, T/2, H/2]), 0, L/2, H/2, L/2, T/2, H/2))  # Top
#     walls.append((create_wall([0, -L/2, H/2], [L/2, T/2, H/2]), 0, -L/2, H/2, L/2, T/2, H/2))  # Bottom
#     walls.append((create_wall([-L/2, 0, H/2], [T/2, L/2, H/2]), -L/2, 0, H/2, T/2, L/2, H/2))  # Left
#     walls.append((create_wall([L/2, 0, H/2], [T/2, L/2, H/2]), L/2, 0, H/2, T/2, L/2, H/2))   # Right



#     # === Inner walls ===
#     inner_defs = [
#         ([-8, 10.5, H/2], [20/6, 0.8, H/2]),
#         ([-8,  7.5, H/2], [20/6, 0.8, H/2]),
#         ([-8,  4.5, H/2], [20/6, 0.8, H/2]),
#         ([-8,  1.5, H/2], [20/6, 0.8, H/2]),
#         ([-8, -1.5, H/2], [20/6, 0.8, H/2]),
#         ([-8, -4.5, H/2], [20/6, 0.8, H/2]),
#         ([-8, -7.5, H/2], [20/6, 0.8, H/2]),
#         ([-8,-10.5, H/2], [20/6, 0.8, H/2]),
#         ([-2,  2,   H/2], [1,    9,   H/2]),
#         ([ 1,  2,   H/2], [1,    9,   H/2]),
#         ([ 4,  2,   H/2], [1,    9,   H/2]),
#         ([ 7,  2,   H/2], [1,    9,   H/2]),
#         ([10,  2,   H/2], [1,    9,   H/2]),
#     ]
#     for center, size in inner_defs:
#         # scale both center & size
#         c = [coord * MAP_SCALE for coord in center]
#         s = [dim   * MAP_SCALE for dim   in size]
#         walls.append((create_wall(c, s), *c, *s))
#         # mark footprint in grid
#         mark_occupied(c[0] - s[0], c[0] + s[0],
#                       c[1] - s[1], c[1] + s[1])

#     # === Cylinders ===
#     cyl_defs = [
#         ([10, -10, 0], 1.5, 1.0),
#         ([ 4, -10, 0], 1.5, 1.0),
#         ([-2, -10, 0], 1.5, 1.0),
#     ]
#     for center, radius, height in cyl_defs:
#         c = [center[0]*MAP_SCALE, center[1]*MAP_SCALE, center[2]]
#         r = radius * MAP_SCALE
#         h = height * MAP_SCALE
#         cylinders.append((create_cylinder(c, r, h), *c, r, h))
#         # mark circular footprint
#         # mark_cylinder_on_grid(grid, (c[0], c[1]), r, resolution, map_size)



#     half_L = L / 2
    
    
#     # for r in range(grid.shape[0]):
#     #     for c in range(grid.shape[1]):
#     #         if grid[r, c] == 1:
#     #             static_obs.append(np.array([
#     #                 r * resolution + resolution/2 - half_L,
#     #                 c * resolution + resolution/2 - half_L
#     #             ]))


#     # return grid
#     for r in range(grid.shape[0]):
#         for c in range(grid.shape[1]):
#             if grid[r, c] == 1:
#                 static_obs.append(
#                   np.array([ r*resolution + resolution/2 - half_size,
#                              c*resolution + resolution/2 - half_size ])
#                 )
#     return grid, static_obs

def create_occupancy_grid(resolution, map_size):
    """
    1) Spawns **all** obstacles (outer walls, inner walls, cylinders) in PyBullet.
    2) Marks their footprints in *one* occupancy‐grid array.
    """
    # 0) prep
    L = map_size * MAP_SCALE
    grid_size = int(L / resolution)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    half = L/2

    def world_to_grid(x, y):
        # x,y in world meters → row,col
        col = int((x + half)/resolution)
        row = int((y + half)/resolution)
        return row, col

    def mark_box(xc, yc, sx, sy):
        r0, c0 = world_to_grid(xc - sx, yc - sy)
        r1, c1 = world_to_grid(xc + sx, yc + sy)
        r0, r1 = sorted([max(0,r0), min(grid_size-1, r1)])
        c0, c1 = sorted([max(0,c0), min(grid_size-1, c1)])
        grid[r0:r1+1, c0:c1+1] = 1

    def mark_circle(xc, yc, radius):
        # simple bounding‐square then per‐cell check
        r0, c0 = world_to_grid(xc - radius, yc - radius)
        r1, c1 = world_to_grid(xc + radius, yc + radius)
        for r in range(max(0,r0), min(grid_size, r1+1)):
            for c in range(max(0,c0), min(grid_size, c1+1)):
                # back to world coords for this cell‐center
                wx = (c + 0.5)*resolution - half
                wy = (r + 0.5)*resolution - half
                if (wx-xc)**2 + (wy-yc)**2 <= radius**2:
                    grid[r,c] = 1

    # 1) OUTER WALLS
    T = 0.2 * MAP_SCALE
    H = 1.0 * MAP_SCALE
    outer = [
      ([0,  half, H/2], [half, T/2, H/2]),    # top
      ([0, -half, H/2], [half, T/2, H/2]),    # bot
      ([-half, 0, H/2], [T/2, half, H/2]),    # left
      ([ half,0, H/2], [T/2, half, H/2]),     # right
    ]
    for (ctr, sz) in outer:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=sz),
            baseVisualShapeIndex  =p.createVisualShape(p.GEOM_BOX, halfExtents=sz, rgbaColor=[.5,.5,.5,1]),
            basePosition=ctr
        )
        # footprint:
        mark_box(ctr[0], ctr[1], sz[0], sz[1])

    # 2) INNER WALLS (list of (center,size) in *unscaled* coords)
    inner_defs = [
        ([-8, 10.5, H/2], [20/6,0.8,H/2]),
        ([-8,  7.5, H/2], [20/6,0.8,H/2]),
        ([-8,  4.5, H/2], [20/6,0.8,H/2]),
        ([-8,  1.5, H/2], [20/6,0.8,H/2]),
        ([-8, -1.5, H/2], [20/6,0.8,H/2]),
        ([-8, -4.5, H/2], [20/6,0.8,H/2]),
        ([-8, -7.5, H/2], [20/6,0.8,H/2]),
        ([-8,-10.5, H/2], [20/6,0.8,H/2]),
        ([-2,  2,   H/2], [1,   9,   H/2]),
        ([ 1,  2,   H/2], [1,   9,   H/2]),
        ([ 4,  2,   H/2], [1,   9,   H/2]),
        ([ 7,  2,   H/2], [1,   9,   H/2]),
        ([10,  2,   H/2], [1,   9,   H/2]),
    ]
    for ctr, sz in inner_defs:
        # apply MAP_SCALE all at once
        c = [ctr[0]*MAP_SCALE, ctr[1]*MAP_SCALE, ctr[2]]
        s = [sz[0]*MAP_SCALE,  sz[1]*MAP_SCALE,  sz[2]]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=s),
            baseVisualShapeIndex  =p.createVisualShape(p.GEOM_BOX, halfExtents=s, rgbaColor=[.5,.5,.5,1]),
            basePosition=c
        )
        mark_box(c[0], c[1], s[0], s[1])

    # 3) CYLINDERS
    cyl_defs = [
      ([10,-10,0], 1.5,1.0),
      ([ 4,-10,0], 1.5,1.0),
      ([-2,-10,0], 1.5,1.0),
    ]
    for ctr, r, h in cyl_defs:
        c = [ctr[0]*MAP_SCALE, ctr[1]*MAP_SCALE, ctr[2]]
        R = r * MAP_SCALE
        Hc = h * MAP_SCALE
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=R, height=Hc),
            baseVisualShapeIndex  =p.createVisualShape(p.GEOM_CYLINDER, radius=R, length=Hc, rgbaColor=[.5,.5,.5,1]),
            basePosition=[c[0],c[1],c[2]+Hc/2]
        )
        mark_circle(c[0], c[1], R)

    # 4) build static_obs list (optional)
    static_obs = []
    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r,c]:
                wx = (c+0.5)*resolution - half
                wy = (r+0.5)*resolution - half
                static_obs.append(np.array([wx,wy]))

    return grid, static_obs



def create_world(client, resolution=0.1, map_size=25.0):
    p.resetSimulation(client)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

    grid, static_obs = create_occupancy_grid(resolution, map_size)
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

    static_obs = []  # Fill this if you need additional static obstacle metadata
    return robot_id, left_wheel_joint_id, right_wheel_joint_id, grid, static_obs