import numpy as np
import heapq
from scipy.ndimage import binary_dilation


def inflate_obstacles(grid, robot_radius_m, grid_resolution_m):
    """
    Inflate binary obstacle map to account for robot radius.
    
    grid: 2D numpy array (0 = free, 1 = obstacle)
    robot_radius_m: radius of robot in meters
    grid_resolution_m: size of 1 grid cell in meters
    """
    # inflate_radius = int(np.ceil(robot_radius_m / grid_resolution_m))
    inflate_radius = max(1, int(np.ceil(0.09 * robot_radius_m / grid_resolution_m)))

    structure = np.ones((2 * inflate_radius + 1, 2 * inflate_radius + 1), dtype=np.uint8)

    # Create binary obstacle mask
    obstacle_mask = (grid == 1)

    # Perform binary dilation
    inflated_mask = binary_dilation(obstacle_mask, structure=structure)

    # Return new grid
    inflated_grid = np.where(inflated_mask, 1, 0)  # force binary result (0 or 1)
    return inflated_grid



def a_star(grid, start, goal):
    """A* path planning algorithm for grid-based navigation."""

    print("A* Grid unique values:", np.unique(grid))

    # Safety checks
    if grid[start[1], start[0]] != 0:
        raise ValueError(f"Start {start} is not in free space!")
    if grid[goal[1], goal[0]] != 0:
        raise ValueError(f"Goal {goal} is not in free space!")

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (-1, -1), (-1, 1), (1, -1)]

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        print(f"\n📌 Expanding node: {current}, cost: {current_cost:.3f}")

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            print("\n✅ Final path:")
            for i, (x, y) in enumerate(path):
                print(f"  Step {i}: Grid[{x}, {y}], value={grid[y, x]}")
            print("Start value:", grid[start[1], start[0]])
            print("Goal value:", grid[goal[1], goal[0]])
            return path

        for dx, dy in neighbors:
            next_node = (current[0] + dx, current[1] + dy)

            # Bounds check
            if not (0 <= next_node[0] < grid.shape[1] and 0 <= next_node[1] < grid.shape[0]):
                continue

            # Obstacle check
            if grid[next_node[1], next_node[0]] != 0:
                continue

            # Prevent diagonal corner-cutting
            if abs(dx) == 1 and abs(dy) == 1:
                n1 = (current[0] + dx, current[1])
                n2 = (current[0], current[1] + dy)
                if (not (0 <= n1[0] < grid.shape[1] and 0 <= n1[1] < grid.shape[0]) or
                    not (0 <= n2[0] < grid.shape[1] and 0 <= n2[1] < grid.shape[0])):
                    continue
                if grid[n1[1], n1[0]] != 0 or grid[n2[1], n2[0]] != 0:
                    continue

            # Cost update
            move_cost = np.hypot(dx, dy)
            new_cost = cost_so_far[current] + move_cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(open_set, (priority, new_cost, next_node))
                came_from[next_node] = current

    # If we reach here, no path was found
    print("❌ No valid path found by A*")
    return []
