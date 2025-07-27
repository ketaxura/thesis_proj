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


    r0, c0 = start      # whatever you passed into a_star
    print(f"Checking start at grid[{r0},{c0}] = {grid[r0,c0]}")
    r1, c1 = goal
    print(f"Checking goal  at grid[{r1},{c1}] = {grid[r1,c1]}")

    # Safety checks
    if grid[start[0], start[1]] != 0:
        raise ValueError(f"Start {start} is not in free space!")
    if grid[goal[0], goal[1]] != 0:
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

        for dr, dc in neighbors:
            nr, nc = (current[0] + dr, current[1] + dc)

            # Bounds check
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue

            # Obstacle check
            if grid[nr, nc] != 0:
                continue

            # Prevent diagonal corner-cutting
            if abs(dr) == 1 and abs(dc) == 1:
                # check the two orthogonal neighbors
                if grid[current[0] + dr, current[1]] != 0 or \
                grid[current[0], current[1] + dc] != 0:
                    continue

                # 4) Cost update & enqueue
            move_cost = np.hypot(dr, dc)
            new_cost = cost_so_far[current] + move_cost
            neighbor = (nr, nc)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    # If we reach here, no path was found
    print("❌ No valid path found by A*")
    return []
