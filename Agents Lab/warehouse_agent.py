"""
Warehouse Navigation: Goal-Based Intelligent Agent Implementation
-----------------------------------------------------------------
Artificial Intelligence: Agents Laboratory Exercise

This module implements an autonomous goal-based agent capable of navigating
a warehouse grid environment, avoiding shelving unit obstacles, and planning
collision-free trajectories between a loading bay (Start) and dispatch area (Goal).

Algorithms Implemented:
1. Breadth-First Search (BFS) - Uninformed optimal search for unit-cost grids.
2. A* Search (A-Star) - Heuristic-guided optimal search using Manhattan distance.
"""

from collections import deque
import heapq
import time
from typing import List, Tuple, Dict, Optional, Set


class WarehouseEnvironment:
    """
    Represents the 2D grid warehouse environment.

    Attributes:
        grid (List[str]): Original character layout of the warehouse.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        start (Tuple[int, int]): Coordinates (row, col) of the loading bay 'S'.
        goal (Tuple[int, int]): Coordinates (row, col) of the dispatch area 'G'.
        obstacles (Set[Tuple[int, int]]): Coordinates of impassable shelving units '#'.
    """

    DEFAULT_MAP = [
        "#####################",
        "#S....#............G#",
        "#.##....##########..#",
        "#....##.............#",
        "#.######.###.#.###..#",
        "#........#..........#",
        "#####################"
    ]

    # Cardinal actions and relative coordinate offsets
    ACTIONS: Dict[str, Tuple[int, int]] = {
        'Up': (-1, 0),
        'Down': (1, 0),
        'Left': (0, -1),
        'Right': (0, 1)
    }

    def __init__(self, map_layout: Optional[List[str]] = None):
        self.grid = map_layout if map_layout is not None else self.DEFAULT_MAP
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start = self._locate_char('S')
        self.goal = self._locate_char('G')
        self.obstacles = self._extract_obstacles()

        if self.start is None:
            raise ValueError("Environment layout must contain a Start position 'S'.")
        if self.goal is None:
            raise ValueError("Environment layout must contain a Goal position 'G'.")

    def _locate_char(self, target: str) -> Optional[Tuple[int, int]]:
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == target:
                    return (r, c)
        return None

    def _extract_obstacles(self) -> Set[Tuple[int, int]]:
        obs = set()
        for r, row in enumerate(self.grid):
            for c, val in enumerate(row):
                if val == '#':
                    obs.add((r, c))
        return obs

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Checks whether state (row, col) is within grid boundaries and not an obstacle."""
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols and state not in self.obstacles

    def get_successors(self, state: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Transition model: returns list of valid (action, successor_state) pairs.
        """
        r, c = state
        successors = []
        for action, (dr, dc) in self.ACTIONS.items():
            next_state = (r + dr, c + dc)
            if self.is_valid_state(next_state):
                successors.append((action, next_state))
        return successors

    def render(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """
        Generates an ASCII visualization of the warehouse, optionally overlaying
        the planned path with '*' symbols.
        """
        path_set = set(path) if path else set()
        rendered_rows = []

        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append('S')
                elif pos == self.goal:
                    row_chars.append('G')
                elif pos in path_set:
                    row_chars.append('*')
                elif pos in self.obstacles:
                    row_chars.append('#')
                else:
                    row_chars.append('.')
            rendered_rows.append("".join(row_chars))

        return "\n".join(rendered_rows)


class GoalBasedAgent:
    """
    A Goal-Based Intelligent Agent designed to achieve navigation objectives
    through explicit search and planning in a world model.
    """

    def __init__(self, env: WarehouseEnvironment, algorithm: str = 'astar'):
        self.env = env
        self.algorithm = algorithm.lower()

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """
        Admissible and consistent heuristic for 4-connected grid pathfinding.
        h(n) = |r1 - r2| + |c1 - c2|
        """
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def plan_path(self) -> Dict[str, any]:
        """
        Executes decision-making search to find a collision-free path from Start to Goal.
        Returns a dictionary containing path, action sequence, nodes expanded, and execution time.
        """
        start_time = time.perf_counter()

        if self.algorithm == 'bfs':
            result = self._breadth_first_search()
        elif self.algorithm in ('astar', 'a*'):
            result = self._a_star_search()
        else:
            raise ValueError(f"Unknown search algorithm: {self.algorithm}")

        elapsed_time_ms = (time.perf_counter() - start_time) * 1000
        result['execution_time_ms'] = elapsed_time_ms
        return result

    def _breadth_first_search(self) -> Dict[str, any]:
        """
        Uninformed search exploring all states level-by-level using a FIFO queue.
        Guarantees shortest path in unweighted unit-cost graphs.
        """
        start = self.env.start
        goal = self.env.goal

        # Queue elements: (current_state, path_so_far, actions_so_far)
        queue = deque([(start, [start], [])])
        visited: Set[Tuple[int, int]] = {start}
        nodes_expanded = 0

        while queue:
            curr, path, actions = queue.popleft()
            nodes_expanded += 1

            if curr == goal:
                return {
                    'found': True,
                    'path': path,
                    'actions': actions,
                    'cost': len(actions),
                    'nodes_expanded': nodes_expanded,
                    'algorithm': 'Breadth-First Search (BFS)'
                }

            for action, successor in self.env.get_successors(curr):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, path + [successor], actions + [action]))

        return {
            'found': False,
            'path': [],
            'actions': [],
            'cost': 0,
            'nodes_expanded': nodes_expanded,
            'algorithm': 'Breadth-First Search (BFS)'
        }

    def _a_star_search(self) -> Dict[str, any]:
        """
        Informed heuristic search using f(n) = g(n) + h(n).
        - g(n): exact cost from start to current state.
        - h(n): estimated cost to goal via Manhattan distance.
        Guarantees optimal path while significantly reducing explored search space.
        """
        start = self.env.start
        goal = self.env.goal

        # Priority Queue entries: (f_score, tie_breaker_counter, g_score, curr_state, path, actions)
        counter = 0
        h_start = self.manhattan_distance(start, goal)
        pq = [(h_start, counter, 0, start, [start], [])]
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        nodes_expanded = 0

        while pq:
            f, _, g, curr, path, actions = heapq.heappop(pq)
            nodes_expanded += 1

            if curr == goal:
                return {
                    'found': True,
                    'path': path,
                    'actions': actions,
                    'cost': len(actions),
                    'nodes_expanded': nodes_expanded,
                    'algorithm': 'A* Search (Manhattan Heuristic)'
                }

            if g > g_scores.get(curr, float('inf')):
                continue

            for action, successor in self.env.get_successors(curr):
                tentative_g = g + 1
                if tentative_g < g_scores.get(successor, float('inf')):
                    g_scores[successor] = tentative_g
                    counter += 1
                    h_score = self.manhattan_distance(successor, goal)
                    f_score = tentative_g + h_score
                    heapq.heappush(pq, (f_score, counter, tentative_g, successor,
                                         path + [successor], actions + [action]))

        return {
            'found': False,
            'path': [],
            'actions': [],
            'cost': 0,
            'nodes_expanded': nodes_expanded,
            'algorithm': 'A* Search (Manhattan Heuristic)'
        }


def main():
    print("=" * 70)
    print("AUTONOMOUS WAREHOUSE NAVIGATION: GOAL-BASED AGENT EVALUATION")
    print("=" * 70)

    # Initialize Environment
    env = WarehouseEnvironment()
    print("\n1. Environment Layout (Dimensions: 7 rows x 21 columns):")
    print(env.render())
    print(f"Start Coordinate (Loading Bay 'S'): {env.start}")
    print(f"Goal Coordinate (Dispatch Area 'G'): {env.goal}")
    print(f"Total Obstacle Cells: {len(env.obstacles)}")

    # -------------------------------------------------------------
    # 2. Planning with A* Search
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. Planning with Informed Agent (A* Heuristic Search)")
    print("-" * 70)
    astar_agent = GoalBasedAgent(env, algorithm='astar')
    astar_result = astar_agent.plan_path()

    if astar_result['found']:
        print(f"Status: Path Successfully Found!")
        print(f"Total Steps (Cost): {astar_result['cost']}")
        print(f"Nodes Expanded: {astar_result['nodes_expanded']}")
        print(f"Planning Time: {astar_result['execution_time_ms']:.3f} ms")
        print(f"Action Sequence ({len(astar_result['actions'])} moves):")
        print("  " + " -> ".join(astar_result['actions']))
        print("\nVisualized Trajectory (* = vehicle path):")
        print(env.render(astar_result['path']))
    else:
        print("Warning: No collision-free path exists to the destination.")

    # -------------------------------------------------------------
    # 3. Planning with Uninformed Search (BFS Comparison)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. Comparative Benchmark: Uninformed Search (BFS)")
    print("-" * 70)
    bfs_agent = GoalBasedAgent(env, algorithm='bfs')
    bfs_result = bfs_agent.plan_path()

    print(f"Algorithm: {bfs_result['algorithm']}")
    print(f"Path Length: {bfs_result['cost']} steps")
    print(f"Nodes Expanded: {bfs_result['nodes_expanded']}")
    print(f"Planning Time: {bfs_result['execution_time_ms']:.3f} ms")

    # Efficiency Comparison
    pruning_efficiency = ((bfs_result['nodes_expanded'] - astar_result['nodes_expanded']) /
                          bfs_result['nodes_expanded']) * 100
    print(f"\nSearch Space Reduction by A*: {pruning_efficiency:.1f}% fewer nodes explored!")

    # -------------------------------------------------------------
    # 4. Edge Case Test: Unreachable Goal (Blocked Path)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. Diagnostic Test: Unreachable Destination (Deadlock Scenario)")
    print("-" * 70)
    # Surround goal (1, 19) with obstacles
    blocked_map = [row for row in env.DEFAULT_MAP]
    # Block left of G: (1, 18), below G: (2, 19)
    blocked_row1 = list(blocked_map[1])
    blocked_row1[18] = '#'
    blocked_map[1] = "".join(blocked_row1)
    blocked_row2 = list(blocked_map[2])
    blocked_row2[19] = '#'
    blocked_map[2] = "".join(blocked_row2)

    blocked_env = WarehouseEnvironment(blocked_map)
    blocked_agent = GoalBasedAgent(blocked_env, algorithm='astar')
    blocked_result = blocked_agent.plan_path()

    if not blocked_result['found']:
        print("Result: Verified! The agent correctly detected that NO path exists.")
        print(f"Nodes explored before terminating: {blocked_result['nodes_expanded']}")

    print("\n" + "=" * 70)
    print("ALL TESTS AND DEMONSTRATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == '__main__':
    main()
