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
        # ← HERE ←
        lidar = obs[:env.num_rays]
        print("LiDAR readings:", np.round(lidar, 3))
        print("––––––––––––––––––––––")

        print(f"Reward: {reward:.3f}, Done: {done}")
        # …
        if done: break
        print(f"Reward: {reward:.3f}, Done: {done}")
        time.sleep(0.05)
        if done:
            print("Episode finished")
            break

    p.disconnect()
