import pybullet as p
import numpy as np
import time
from .utils import MAP_SCALE
import os
import trimesh as tm
import numpy as np

# === OBJ map config ===
USE_OBJ_MAP = True                     # flip to False to use the old box/cyl map
MAP_DIR     = "/home/max/thesis_proj/parkour_maps"
OBJ_NAME    = "parkour.obj"
CAD_IN_MM   = True                     # OBJ exported in millimeters?
SCALE       = 0.001 if CAD_IN_MM else 1.0
VERT_NZ_THR = 0.20                     # keep |nz|<0.2 ⇒ vertical-ish faces

# === Optional spawn bounding boxes for OBJ map (world meters) ===
# Rectangle format: (x_min, y_min, x_max, y_max)
SPAWN_BBOX_START = None  # e.g., (9.0, -11.5, 11.5, 11.5)  # right-most bay
SPAWN_BBOX_GOAL  = None  # e.g., (-11.5, -11.5, -9.0, 11.5)  # left side

DEBUG_DRAW_INFLATION = True   # turn off when you’re done
INFLATE_RADIUS = 0.18         # what you currently use; try 0.12 if doorways seal



# Tip: fill these only when USE_OBJ_MAP is True
if USE_OBJ_MAP:
    # TODO: tune these numbers to your map; they’re world meters
    SPAWN_BBOX_START = (-4.9, 8.6, 4.9, 9.9)   # <- example: right-most space
    SPAWN_BBOX_GOAL  = (-4.9, -9.9, 4.9, -8.1) # <- example: left column
    # SPAWN_BBOX_GOAL  = (-11.5, -11.5, -9.0, 11.5) # <- example: left column



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
      ([0,  half*3, H/2], [half*1.25, T/2, H/2]),    # top
      ([0, -half*3, H/2], [half*1.25, T/2, H/2]),    # bot
      ([-half*1.25, 0, H/2], [T/2, half*3, H/2]),    # left
      ([ half*1.25 ,0, H/2], [T/2, half*3, H/2]),     # right
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



#red is X
#green is Y
#blue is Z

    # 2) INNER WALLS (list of (center,size) in *unscaled* coords)
    inner_defs = [
        
        ([-13.5, 30, H/2], [2,4,H/2]),
        ([-8, 30, H/2], [2,2,H/2]),
        
        
        ([13.5, 30, H/2], [2,4,H/2]),
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
      ([18,-18,0], 0.4,1.0),
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



def _draw_grid_overlay_binary(mask, meta, color=(1,0,0), z=0.02, stride=1, lineWidth=1):
    """
    Draw wireframe rectangles for True cells in 'mask' at height z.
    meta = ((origin_x, origin_y), (size_x, size_y), res) from _mesh_to_occupancy
    """
    (ox, oy), (_, _), res = meta
    ny, nx = mask.shape
    for r in range(0, ny, stride):
        cols = np.where(mask[r])[0]
        y0 = oy + r * res
        for c in cols:
            x0 = ox + c * res
            # corners of this cell
            p0 = [x0,       y0,       z]
            p1 = [x0+res,   y0,       z]
            p2 = [x0+res,   y0+res,   z]
            p3 = [x0,       y0+res,   z]
            p.addUserDebugLine(p0, p1, color, lineWidth=lineWidth, lifeTime=0)
            p.addUserDebugLine(p1, p2, color, lineWidth=lineWidth, lifeTime=0)
            p.addUserDebugLine(p2, p3, color, lineWidth=lineWidth, lifeTime=0)
            p.addUserDebugLine(p3, p0, color, lineWidth=lineWidth, lifeTime=0)

def _draw_occupancy_layers(raw_grid, raw_meta, inf_grid, inf_meta, stride=1):
    # raw (no inflation): blue
    _draw_grid_overlay_binary(raw_grid.astype(bool), raw_meta, color=(0,0,1), z=0.015, stride=stride)
    # inflated: red
    _draw_grid_overlay_binary(inf_grid.astype(bool), inf_meta, color=(1,0,0), z=0.030, stride=stride)
    # added-by-inflation: orange
    added = (inf_grid == 1) & (raw_grid == 0)
    _draw_grid_overlay_binary(added, inf_meta, color=(1.0, 0.6, 0.0), z=0.050, stride=stride, lineWidth=2)





def _load_vertical_from_obj(obj_path, scale=SCALE, nz_thr=VERT_NZ_THR, recenter_xy=True):
    """
    Load OBJ, keep vertical faces only, return (V,F, bounds) in meters.
    bounds = ((xmin,ymin),(xmax,ymax)) after optional recenter.
    """
    m = tm.load(obj_path, force='mesh')
    if isinstance(m, tm.Scene):
        m = tm.util.concatenate([g for g in m.geometry.values()])
    m.remove_unreferenced_vertices()

    nz = m.face_normals[:, 2]
    keep = np.abs(nz) < nz_thr
    m.update_faces(keep)
    m.remove_unreferenced_vertices()

    V = (m.vertices * scale).astype(np.float32)
    F = m.faces.astype(np.int32)
    if V.size == 0 or F.size == 0:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.int32), ((-5,-5),(5,5))

    # optional recenter in XY
    xmin, ymin, _ = V.min(axis=0); xmax, ymax, _ = V.max(axis=0)
    if recenter_xy:
        cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        V[:,0] -= cx; V[:,1] -= cy
        xmin -= cx; xmax -= cx; ymin -= cy; ymax -= cy

    return V, F, ((float(xmin), float(ymin)), (float(xmax), float(ymax)))


def _load_env_from_obj(client, map_dir, obj_name, scale=SCALE):
    """Collision from memory (vertical faces), visual directly from OBJ."""
    obj_path = os.path.join(map_dir, obj_name)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Missing OBJ: {obj_path}")

    V, F, bounds = _load_vertical_from_obj(obj_path, scale=scale)
    # visual straight from file (mtl respected), collision from memory mesh
    col_id = p.createCollisionShape(p.GEOM_MESH,
                                    vertices=V.tolist(),
                                    indices=F.flatten().tolist(),
                                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_id = p.createVisualShape(
    p.GEOM_MESH,
    fileName=obj_path,
    meshScale=[scale]*3,
    rgbaColor=[0.80, 0.8, 0.85, 1.0],   # color [0.95, 0.95, 0.98, 1.0])
    
    )

    env_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_id,
                               baseVisualShapeIndex=vis_id, basePosition=[0,0,0])
    p.setCollisionFilterGroupMask(env_id, -1, 1, 1)
    p.changeDynamics(env_id, -1, lateralFriction=1.0, restitution=0.0)
    return env_id, V, F, bounds


def _mesh_to_occupancy(V, F, resolution=0.10, dilate=0.18, margin=0.50):
    """
    Convert vertical-face mesh edges to a 2D occupancy grid.
    Returns (grid, static_obs, meta), meta = ((origin_x,origin_y),(size_x,size_y),res)
    """
    if V.size == 0 or F.size == 0:
        n = max(4, int(10.0/resolution))
        return np.zeros((n,n), np.uint8), [], ((-5,-5),(10,10), resolution)

    xmin, ymin = V[:,0].min()-margin, V[:,1].min()-margin
    xmax, ymax = V[:,0].max()+margin, V[:,1].max()+margin
    size_x, size_y = xmax-xmin, ymax-ymin
    nx, ny = int(np.ceil(size_x/resolution)), int(np.ceil(size_y/resolution))
    grid = np.zeros((ny, nx), dtype=np.uint8)

    def w2g(x, y):
        c = int((x - xmin)/resolution); r = int((y - ymin)/resolution)
        return r, c

    # edge set
    edges = set()
    for tri in F:
        i, j, k = map(int, tri)
        for a,b in ((i,j),(j,k),(k,i)):
            if a > b: a,b = b,a
            edges.add((a,b))

    # rasterize with simple dilation
    rad = max(1, int(np.ceil(dilate/resolution)))
    for a,b in edges:
        x0,y0 = V[a,0], V[a,1]; x1,y1 = V[b,0], V[b,1]
        steps = max(2, int(np.hypot(x1-x0, y1-y0)/(0.25*resolution)))
        for t in np.linspace(0,1,steps):
            xx = x0*(1-t) + x1*t; yy = y0*(1-t) + y1*t
            r,c = w2g(xx,yy)
            for dr in range(-rad,rad+1):
                for dc in range(-rad,rad+1):
                    rr,cc = r+dr, c+dc
                    if 0 <= rr < ny and 0 <= cc < nx:
                        grid[rr,cc] = 1

    static_obs = []
    for r in range(ny):
        cols = np.where(grid[r] != 0)[0]
        wy = ymin + (r+0.5)*resolution
        for c in cols:
            wx = xmin + (c+0.5)*resolution
            static_obs.append(np.array([wx,wy], dtype=np.float32))

    meta = ((float(xmin), float(ymin)), (float(size_x), float(size_y)), float(resolution))
    return grid, static_obs, meta



def create_world(client, resolution=0.1, map_size=25.0):
    p.resetSimulation(client)
    p.setGravity(0, 0, -9.81)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # floor (no checker)
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, textureUniqueId=-1)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.68, 1.0])

    if USE_OBJ_MAP:
        env_id, V, F, _ = _load_env_from_obj(client, MAP_DIR, OBJ_NAME, scale=SCALE)
        # force flat color if .mtl is strong
        # p.changeVisualShape(env_id, -1, textureUniqueId=-1)

        grid, static_obs, meta = _mesh_to_occupancy(V, F, resolution=resolution, dilate=0.1, margin=0.50)

        robot_id = p.loadURDF(
            "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
            basePosition=[1.0, 2.0, 0.1], useFixedBase=False
        )
    else:
        grid, static_obs = create_occupancy_grid(resolution, map_size)
        robot_id = p.loadURDF(
            "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
            basePosition=[1.0*MAP_SCALE, 2.0*MAP_SCALE, 0.1], useFixedBase=False
        )

    # wheels
    left_wheel_joint_id = right_wheel_joint_id = None
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[1].decode()
        if "wheel_left_joint"  in name: left_wheel_joint_id  = i
        if "wheel_right_joint" in name: right_wheel_joint_id = i
    assert left_wheel_joint_id is not None and right_wheel_joint_id is not None

    for _ in range(120):
        p.stepSimulation(); time.sleep(1.0/60.0)

    return robot_id, left_wheel_joint_id, right_wheel_joint_id, grid, static_obs

    # reset + gravity + data path for plane.urdf
    p.resetSimulation(client)
    p.setGravity(0, 0, -9.81)
    import pybullet_data
    # world.py  (inside create_world, after p.resetSimulation/p.setGravity)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF("plane.urdf")          # one plane only
    p.changeVisualShape(plane_id, -1, textureUniqueId=-1)          # remove checker
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.95, 0.95, 0.98, 1.0])  # flat color

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    if USE_OBJ_MAP:
        env_id, V, F, bounds = _load_env_from_obj(client, MAP_DIR, OBJ_NAME, scale=SCALE)

        # Inflated grid = what your planner uses now
        grid, static_obs, meta = _mesh_to_occupancy(
            V, F, resolution=resolution, dilate=0.0, margin=0.50
        )

        # # Optional: draw layers to see chokepoints
        # if DEBUG_DRAW_INFLATION:
        #     grid_raw, _, meta_raw = _mesh_to_occupancy(
        #         V, F, resolution=resolution, dilate=0.0, margin=0.50
        #     )
        #     _draw_occupancy_layers(grid_raw, meta_raw, grid, meta, stride=1)  # increase stride to 2/3 if slow

        # Spawn TB3 (meters)
        robot_id = p.loadURDF(
            "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
            basePosition=[1.0, 2.0, 0.1],
            useFixedBase=False
        )


    else:
        # --- Your existing parametric map path (unchanged) ---
        grid, static_obs = create_occupancy_grid(resolution, map_size)

        # Keep your existing scaling for this path
        robot_id = p.loadURDF(
            "/home/max/thesis_proj/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
            basePosition=[1.0 * MAP_SCALE, 2.0 * MAP_SCALE, 0.1],
            useFixedBase=False
        )

    # Find wheel joints (unchanged)
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

    # Let things settle briefly
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    return robot_id, left_wheel_joint_id, right_wheel_joint_id, grid, static_obs

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