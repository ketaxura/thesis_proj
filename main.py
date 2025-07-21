from crowdnav.env import CrowdNavPyBulletEnv
import numpy as np
import time
import pybullet as p

if __name__ == "__main__":
    env = CrowdNavPyBulletEnv()
    obs = env.reset()
    print("Environment reset complete")

    while True:
        action = np.array([1.0, 0.1, 0.1])  # Example weights for goal/smooth/obs
        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward:.3f}, Done: {done}")
        time.sleep(0.05)
        if done:
            print("Episode finished")
            break

    p.disconnect()
