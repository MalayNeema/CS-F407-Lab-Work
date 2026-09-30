import os
import matplotlib.pyplot as plt
import numpy as np
import heapq
from collections import deque

save_dir = r"C:\Users\nikhi\.gemini\antigravity\scratch\warehouse_agent_lab"
figs_dir = os.path.join(save_dir, "figures")
os.makedirs(figs_dir, exist_ok=True)

warehouse_map = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]

rows = len(warehouse_map)
cols = len(warehouse_map[0])

# Parse grid
grid_array = np.zeros((rows, cols))
start = None
goal = None

for r in range(rows):
    for c in range(cols):
        char = warehouse_map[r][c]
        if char == '#':
            grid_array[r, c] = 1 # Obstacle
        elif char == 'S':
            start = (r, c)
        elif char == 'G':
            goal = (r, c)

# Actions
ACTIONS = {
    'Up': (-1, 0),
    'Down': (1, 0),
    'Left': (0, -1),
    'Right': (0, 1)
}

def get_neighbors(pos):
    r, c = pos
    neighbors = []
    for action, (dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid_array[nr, nc] == 0:
            neighbors.append((action, (nr, nc)))
    return neighbors

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Run BFS
def run_bfs():
    queue = deque([(start, [start], [])])
    visited = {start}
    order_explored = []
    
    while queue:
        curr, path, acts = queue.popleft()
        order_explored.append(curr)
        if curr == goal:
            return path, acts, order_explored
        for act, neighbor in get_neighbors(curr):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], acts + [act]))
    return None, None, order_explored

# Run A*
def run_astar():
    pq = [(manhattan_distance(start, goal), 0, start, [start], [])]
    g_costs = {start: 0}
    order_explored = []
    
    while pq:
        f, g, curr, path, acts = heapq.heappop(pq)
        order_explored.append(curr)
        if curr == goal:
            return path, acts, order_explored
        if g > g_costs.get(curr, float('inf')):
            continue
        for act, neighbor in get_neighbors(curr):
            tentative_g = g + 1
            if tentative_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(pq, (f_score, tentative_g, neighbor, path + [neighbor], acts + [act]))
    return None, None, order_explored

bfs_path, bfs_acts, bfs_explored = run_bfs()
astar_path, astar_acts, astar_explored = run_astar()

print(f"BFS Path Length: {len(bfs_path)-1}, Explored: {len(bfs_explored)}")
print(f"A* Path Length: {len(astar_path)-1}, Explored: {len(astar_explored)}")

# -------------------------------------------------------------------------
# Figure 1: Warehouse Map & Problem Layout
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
# Custom colormap for grid: 0=free (white), 1=obstacle (dark slate)
display_mat = grid_array.copy()
ax.imshow(display_mat, cmap='binary', origin='upper', alpha=0.3)

# Grid lines
ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.8)
ax.tick_params(which='minor', size=0)

# Draw obstacles as blocks
for r in range(rows):
    for c in range(cols):
        if grid_array[r, c] == 1:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='#2C3E50', edgecolor='gray'))

# Mark Start and Goal
ax.plot(start[1], start[0], 's', color='#27AE60', markersize=16, label='Start (S) - Loading Bay')
ax.text(start[1], start[0], 'S', color='white', weight='bold', ha='center', va='center', fontsize=11)

ax.plot(goal[1], goal[0], '*', color='#E74C3C', markersize=20, label='Goal (G) - Dispatch Area')
ax.text(goal[1], goal[0]+0.3, 'G', color='#C0392B', weight='bold', ha='center', va='top', fontsize=12)

ax.set_title('Warehouse Environment Layout (7 x 21 Grid World)', fontsize=13, weight='bold', pad=12)
ax.set_xlabel('Column Coordinate (X)', fontsize=11)
ax.set_ylabel('Row Coordinate (Y)', fontsize=11)
ax.set_xticks(range(0, cols, 2))
ax.set_yticks(range(rows))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=True, fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, "fig1_warehouse_map.png"))
fig.savefig(os.path.join(save_dir, "fig1_warehouse_map.png"))
plt.close()
print("Saved fig1_warehouse_map.png")

# -------------------------------------------------------------------------
# Figure 2: Search Frontier Comparison (BFS vs A*)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=300)

for ax, explored, title, count in [(ax1, bfs_explored, "Breadth-First Search (Uninformed)", len(bfs_explored)),
                                   (ax2, astar_explored, "A* Search (Manhattan Distance Heuristic)", len(astar_explored))]:
    # Base obstacles
    for r in range(rows):
        for c in range(cols):
            if grid_array[r, c] == 1:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='#2C3E50', edgecolor='gray'))
            else:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='#ECF0F1', edgecolor='#BDC3C7'))

    # Mark explored nodes
    for i, (er, ec) in enumerate(explored):
        if (er, ec) not in (start, goal):
            ax.add_patch(plt.Rectangle((ec-0.5, er-0.5), 1, 1, facecolor='#F39C12', alpha=0.65, edgecolor='#D68910'))

    # Mark optimal path
    path_c = [c for r, c in astar_path]
    path_r = [r for r, c in astar_path]
    ax.plot(path_c, path_r, color='#2980B9', linewidth=3, linestyle='-', marker='o', markersize=5, label='Optimal Path (20 steps)')

    # Start & Goal
    ax.plot(start[1], start[0], 's', color='#27AE60', markersize=14)
    ax.text(start[1], start[0], 'S', color='white', weight='bold', ha='center', va='center', fontsize=9)
    ax.plot(goal[1], goal[0], '*', color='#E74C3C', markersize=18)
    ax.text(goal[1], goal[0], 'G', color='white', weight='bold', ha='center', va='center', fontsize=9)

    ax.set_title(f"{title}: {count} States Explored", fontsize=11, weight='bold')
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

ax1.legend(loc='lower right', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, "fig2_search_comparison.png"))
fig.savefig(os.path.join(save_dir, "fig2_search_comparison.png"))
plt.close()
print("Saved fig2_search_comparison.png")

# -------------------------------------------------------------------------
# Figure 3: Optimal Path Navigation Overlay
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
for r in range(rows):
    for c in range(cols):
        if grid_array[r, c] == 1:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='#34495E', edgecolor='gray'))
        else:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, facecolor='#FFFFFF', edgecolor='#BDC3C7'))

# Path line
path_c = [c for r, c in astar_path]
path_r = [r for r, c in astar_path]
ax.plot(path_c, path_r, color='#E67E22', linewidth=3.5, label='Agent Trajectory (20 moves)')

# Directional arrows
for i in range(len(astar_path) - 1):
    r1, c1 = astar_path[i]
    r2, c2 = astar_path[i+1]
    ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                arrowprops=dict(arrowstyle="->", color='#D35400', lw=2, mutation_scale=12))

ax.plot(start[1], start[0], 's', color='#27AE60', markersize=16, label='Start S (1, 1)')
ax.text(start[1], start[0], 'S', color='white', weight='bold', ha='center', va='center', fontsize=10)
ax.plot(goal[1], goal[0], '*', color='#E74C3C', markersize=20, label='Goal G (1, 19)')
ax.text(goal[1], goal[0], 'G', color='white', weight='bold', ha='center', va='center', fontsize=10)

ax.set_title('Goal-Based Agent Optimal Navigation Plan (Collision-Free Trajectory)', fontsize=12, weight='bold', pad=10)
ax.set_xlim(-0.5, cols - 0.5)
ax.set_ylim(rows - 0.5, -0.5)
ax.set_aspect('equal')
ax.set_xticks(range(0, cols, 2))
ax.set_yticks(range(rows))
ax.grid(False)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=True, fontsize=9.5)
plt.tight_layout()
fig.savefig(os.path.join(figs_dir, "fig3_path_solution.png"))
fig.savefig(os.path.join(save_dir, "fig3_path_solution.png"))
plt.close()
print("Saved fig3_path_solution.png")
