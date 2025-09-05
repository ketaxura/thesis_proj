# crowdnav/dynamics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pybullet as p
from .config import TB3
from .kinematics import v_omega_to_wheels, clamp_wheels

@dataclass(frozen=True)
class RobotHandles:
    body: int
    left_joint: int
    right_joint: int
    left_link: int | None = None
    right_link: int | None = None

class Dynamics:
    def __init__(
        self,
        robot_id: int,
        left_joint_id: int,
        right_joint_id: int,
        motor_force: float,
        left_link_id: int | None = None,
        right_link_id: int | None = None,
        cache_fixed_dt: bool = True,
    ):
        # ids
        self.robot_id   = int(robot_id)
        self.left_joint = int(left_joint_id)
        self.right_joint= int(right_joint_id)

        # handles (fallback link indices = joint indices if not given)
        self.h = RobotHandles(
            body=self.robot_id,
            left_joint=self.left_joint,
            right_joint=self.right_joint,
            left_link=left_link_id if left_link_id is not None else left_joint_id,
            right_link=right_link_id if right_link_id is not None else right_joint_id,
        )

        # actuation
        self.motor_force = float(motor_force)

        # physics dt cache (optional)
        self._fixed_dt = (
            float(p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0))
            if cache_fixed_dt else None
        )

    # -------------------------------------------------
    # time-step utilities
    # -------------------------------------------------
    @property
    def fixed_dt(self) -> float:
        if self._fixed_dt is None:
            self._fixed_dt = float(p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0))
        return self._fixed_dt

    def step(self, T: float) -> None:
        """Advance physics for exactly T seconds."""
        n = max(1, int(np.ceil(float(T) / self.fixed_dt)))
        for _ in range(n):
            p.stepSimulation()

    # -------------------------------------------------
    # actuation
    # -------------------------------------------------
    def set_wheel_omegas(self, wL: float, wR: float) -> None:
        p.setJointMotorControl2(self.h.body, self.h.left_joint,
                                p.VELOCITY_CONTROL, targetVelocity=float(wL),
                                force=self.motor_force)
        p.setJointMotorControl2(self.h.body, self.h.right_joint,
                                p.VELOCITY_CONTROL, targetVelocity=float(wR),
                                force=self.motor_force)

    def set_twist(self, v: float, omega: float, tb3: TB3) -> tuple[float, float]:
        wL, wR = v_omega_to_wheels(v, omega, tb3)
        wL, wR = clamp_wheels(wL, wR, tb3)
        self.set_wheel_omegas(wL, wR)
        return wL, wR

    # -------------------------------------------------
    # state access
    # -------------------------------------------------
    def get_pose(self) -> tuple[np.ndarray, float]:
        pos, orn = p.getBasePositionAndOrientation(self.h.body)
        x, y = pos[:2]
        th = p.getEulerFromQuaternion(orn)[2]
        th = (th + np.pi) % (2 * np.pi) - np.pi
        return np.array([x, y], dtype=float), float(th)

    def get_pose_full(self):
        pos, quat = p.getBasePositionAndOrientation(self.h.body)
        R = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
        return np.array(pos, dtype=float), np.array(quat, dtype=float), R

    # -------------------------------------------------
    # housekeeping
    # -------------------------------------------------
    def reset_pose(self, xy: np.ndarray, theta: float, z: float = 0.008) -> None:
        quat = p.getQuaternionFromEuler([0.0, 0.0, float(theta)])
        p.resetBasePositionAndOrientation(self.h.body, [float(xy[0]), float(xy[1]), float(z)], quat)
        p.resetBaseVelocity(self.h.body, [0, 0, 0], [0, 0, 0])

    def get_wheel_speeds(self) -> tuple[float, float]:
        l = p.getJointState(self.h.body, self.h.left_joint)[1]
        r = p.getJointState(self.h.body, self.h.right_joint)[1]
        return float(l), float(r)

    def tune_wheels(
        self,
        lateral_friction=0.6,
        rolling_friction=0.0,
        spinning_friction=0.0,
        linear_damping=0.02,
        angular_damping=0.02,
    ):
        for link in (self.h.left_link, self.h.right_link):
            p.changeDynamics(
                self.h.body, linkIndex=int(link),
                lateralFriction=float(lateral_friction),
                rollingFriction=float(rolling_friction),
                spinningFriction=float(spinning_friction),
                linearDamping=float(linear_damping),
                angularDamping=float(angular_damping),
            )

    def get_contacts(self, other_body: int | None = None):
        return (
            p.getContactPoints(bodyA=self.h.body)
            if other_body is None
            else p.getContactPoints(bodyA=self.h.body, bodyB=other_body)
        )


