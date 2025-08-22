from crowdnav import CrowdNavPyBulletEnv
import numpy as np, pybullet as p

if __name__ == "__main__":
    env = CrowdNavPyBulletEnv()
    obs = env.reset()
    while True:
        obs, reward, done, _ = env.step(None)
        if done: break
    p.disconnect()
