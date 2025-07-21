import numpy as np
import pybullet as p

def create_random_pedestrian(robot_pos, client):
    """Create a random pedestrian with a position and goal."""
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(1.0, 5.0)
    pos = robot_pos + radius * np.array([np.cos(angle), np.sin(angle)])
    goal = np.random.uniform(0, 5, size=(2,))
    sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.1)
    ped_id = p.createMultiBody(
        baseMass=1,
        baseVisualShapeIndex=sphere,
        basePosition=[*pos, 0.1]
    )
    return {'pos': pos, 'goal': goal, 'vel': np.zeros(2), 'id': ped_id}

def update_pedestrian(ped, peds, robot_pos):
    """Update pedestrian position based on goal and repulsion forces."""
    desired_vel = ped['goal'] - ped['pos']
    dist_to_goal = np.linalg.norm(desired_vel)
    if dist_to_goal < 0.2:
        ped['goal'] = np.random.uniform(0, 5, size=(2,))
        return
    preferred_speed = 0.06
    desired_vel = preferred_speed * desired_vel / (dist_to_goal + 1e-6)

    repulsion = np.zeros(2)
    for other in peds:
        if other is not ped:
            diff = ped['pos'] - other['pos']
            d = np.linalg.norm(diff)
            if d < 0.5 and d > 1e-4:
                repulsion += 0.01 * diff / (d**2)

    diff_robot = ped['pos'] - robot_pos
    d_robot = np.linalg.norm(diff_robot)
    if d_robot < 0.5 and d_robot > 1e-4:
        repulsion += 0.02 * diff_robot / (d_robot**2)

    noise = np.random.normal(0, 0.01, size=2)
    ped['vel'] = desired_vel + repulsion + noise
    ped['pos'] += ped['vel']
    p.resetBasePositionAndOrientation(ped['id'], [*ped['pos'], 0.1], [0, 0, 0, 1])