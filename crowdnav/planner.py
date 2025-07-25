import numpy as np
import heapq

def a_star(grid, start, goal):
    """A* path planning algorithm for grid-based navigation."""

    print("A* Grid unique values:", np.unique(grid))
    
    # Optional safety check
    if grid[start] != 0:
        raise ValueError(f"Start {start} is not in free space!")
    if grid[goal] != 0:
        raise ValueError(f"Goal {goal} is not in free space!")

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (-1, 1), (1, -1)]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        print(f"\n📌 Expanding node: {current}, current cost: {current_cost}")

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
        
            next_node = (current[0] + dx, current[1] + dy)
            print(f"    Checking neighbor: {next_node}")

            # Bounds check
            if not (0 <= next_node[0] < grid.shape[0] and 0 <= next_node[1] < grid.shape[1]):
                print("    ⛔ Skipped (out of bounds)")
                continue

            # Obstacle check
            if grid[next_node] == 1:
                print(f"    ⛔ Skipped (obstacle at {next_node})")
                continue

            # Diagonal cutting check
            if abs(dx) == 1 and abs(dy) == 1:
                neighbor1 = (current[0] + dx, current[1])
                neighbor2 = (current[0], current[1] + dy)
                if (grid[neighbor1] == 1 or grid[neighbor2] == 1):
                    print(f"    ⛔ Skipped (corner cutting at {neighbor1} or {neighbor2})")
                    continue

            move_cost = np.hypot(dx, dy)
            new_cost = cost_so_far[current] + move_cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                print(f"    ✅ Adding to open set: {next_node} with new cost: {new_cost:.3f}")

                    
            # Check bounds
            if 0 <= next_node[0] < grid.shape[0] and 0 <= next_node[1] < grid.shape[1]:

                # Skip if it's an obstacle
                if grid[next_node] == 1:
                    continue

                # Prevent corner cutting through diagonals
                if abs(dx) == 1 and abs(dy) == 1:
                    neighbor1 = (current[0] + dx, current[1])   # horizontal neighbor
                    neighbor2 = (current[0], current[1] + dy)   # vertical neighbor
                    if (0 <= neighbor1[0] < grid.shape[0] and 0 <= neighbor1[1] < grid.shape[1] and grid[neighbor1] == 1) or \
                    (0 <= neighbor2[0] < grid.shape[0] and 0 <= neighbor2[1] < grid.shape[1] and grid[neighbor2] == 1):
                        continue

                move_cost = np.hypot(dx, dy)
                new_cost = cost_so_far[current] + move_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_node))
                    came_from[next_node] = current

    print("\n✅ Final path:")
    for i, (y, x) in enumerate(path):
        print(f"  Step {i}: Grid[{y}, {x}], value={grid[y,x]}")

    return []
