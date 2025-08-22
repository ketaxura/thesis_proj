# crowdnav/sensors.py
import numpy as np
import pybullet as p
from .config import LidarCfg

class LidarSensor:
    def __init__(self, cfg: LidarCfg):
        fov = np.deg2rad(cfg.fov_deg)
        ang = np.linspace(-fov/2, fov/2, cfg.num_rays)
        self.dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
        self.max_range = cfg.max_range
        self.height = cfg.height

    def scan(self, body_id: int):
        pos, q = p.getBasePositionAndOrientation(body_id)
        pos = np.array(pos); pos[2] = self.height
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
        ori = np.repeat(pos[None,:], self.dirs.shape[0], axis=0)
        ray_to = ori + self.max_range * (R @ self.dirs.T).T
        hits = p.rayTestBatch(ori.tolist(), ray_to.tolist())
        # return normalized distances ∈ [0,1]
        return np.array([h[2] for h in hits], dtype=np.float32)
    