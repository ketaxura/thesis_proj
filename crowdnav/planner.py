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
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in neighbors:
            next_node = (current[0] + dx, current[1] + dy)
            if 0 <= next_node[0] < grid.shape[0] and 0 <= next_node[1] < grid.shape[1]:
                if grid[next_node] == 1:
                    print(f"Skipping obstacle at {next_node}")
                    continue
                move_cost = np.hypot(dx, dy)
                new_cost = cost_so_far[current] + move_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_node))
                    came_from[next_node] = current

    return []
