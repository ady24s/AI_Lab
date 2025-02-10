import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def compute_levels(tree_edges, start_node):
    """
    Given a list of tree_edges (each as (parent, child)) and a start_node,
    compute the level (distance from the start) for each node in the BFS tree.
    """
    # Build a directed tree from the edge list.
    T = nx.DiGraph()
    T.add_edges_from(tree_edges)
    levels = {start_node: 0}
    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        for child in T.successors(node):
            if child not in levels:
                levels[child] = levels[node] + 1
                queue.append(child)
    return levels

def compute_positions(levels):
    """
    Given a dictionary mapping each node to its level,
    compute a position (x, y) for each node so that:
      - y is determined by the level (with level 0 at the top),
      - nodes in the same level are evenly spaced along x.
    """
    level_nodes = {}
    for node, lvl in levels.items():
        level_nodes.setdefault(lvl, []).append(node)
        
    pos = {}
    for lvl, nodes in level_nodes.items():
        n = len(nodes)
        # Spread nodes evenly from x=0 to x=1.
        for i, node in enumerate(sorted(nodes)):
            # if only one node, center it; otherwise, space them evenly.
            x = 0.5 if n == 1 else (i + 1) / (n + 1)
            pos[node] = (x, -lvl)  # negative y so that level 0 is at the top
    return pos

class GraphVisualization:
    def __init__(self):
        self.edges = []  # list of (node, neighbor)
    
    def add_edge(self, a, b):
        self.edges.append((a, b))
    
    def visualize(self, original_edges, traversal_tree_recursive=None, traversal_tree_non_recursive=None, start_node=None):
        """
        Visualize the original graph (left) plus the BFS traversal trees (center/right)
        using a level-by-level layout computed from the BFS tree.
        """
        # --- Original Graph (using a spring layout) ---
        original_graph = nx.Graph()
        original_graph.add_edges_from(original_edges)
        
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        axes[0].set_title("Original Graph")
        nx.draw(original_graph, ax=axes[0],
                with_labels=True, node_color='lightblue',
                node_size=500, edge_color='gray')
        
        # --- Recursive BFS Traversal Tree ---
        if traversal_tree_recursive:
            rec_graph = nx.DiGraph()
            rec_graph.add_edges_from(traversal_tree_recursive)
            axes[1].set_title("Recursive BFS Traversal")
            if start_node:
                # Compute a level-by-level layout from the start node.
                levels = compute_levels(traversal_tree_recursive, start_node)
                pos = compute_positions(levels)
            else:
                pos = nx.spring_layout(rec_graph)
            nx.draw(rec_graph, pos=pos, ax=axes[1],
                    with_labels=True, node_color='lightgreen',
                    node_size=500, edge_color='black', arrows=True)
        
        # --- Non-Recursive BFS Traversal Tree ---
        if traversal_tree_non_recursive:
            nonrec_graph = nx.DiGraph()
            nonrec_graph.add_edges_from(traversal_tree_non_recursive)
            axes[2].set_title("Non-Recursive BFS Traversal")
            if start_node:
                levels = compute_levels(traversal_tree_non_recursive, start_node)
                pos = compute_positions(levels)
            else:
                pos = nx.spring_layout(nonrec_graph)
            nx.draw(nonrec_graph, pos=pos, ax=axes[2],
                    with_labels=True, node_color='lightcoral',
                    node_size=500, edge_color='black', arrows=True)
        
        plt.tight_layout()
        plt.show()

def bfs_recursive(graph, current_level, visited, tree_edges, enqueued):
    """
    Recursive BFS that processes one level at a time.
    
    Parameters:
    - graph: dictionary mapping a node to a list of its neighbors.
    - current_level: list of nodes in the current level.
    - visited: set of nodes already visited.
    - tree_edges: list to record (parent, child) pairs.
    - enqueued: set of nodes that have been added for processing.
    """
    if not current_level:
        return
    next_level = []
    for node in current_level:
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
        for neighbor in graph.get(node, []):
            if neighbor not in visited and neighbor not in enqueued:
                tree_edges.append((node, neighbor))
                next_level.append(neighbor)
                enqueued.add(neighbor)
    bfs_recursive(graph, next_level, visited, tree_edges, enqueued)

def bfs_non_recursive(graph, start_node):
    """
    Non-recursive BFS that uses a queue. Here, neighbors are processed in reversed order
    to yield a different ordering from the recursive version.
    
    Returns:
    - tree_edges: list of (parent, child) pairs discovered during traversal.
    """
    visited = set()
    queue = deque([start_node])
    enqueued = {start_node}
    tree_edges = []
    while queue:
        node = queue.popleft()
        visited.add(node)
        print(node, end=" ")
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited and neighbor not in enqueued:
                tree_edges.append((node, neighbor))
                queue.append(neighbor)
                enqueued.add(neighbor)
    print()
    return tree_edges

def main():
    file_path = "/home/adyasha/Desktop/AI LAB/dfs.csv"
    data = pd.read_csv(file_path)
    
    graph = {}
    gv = GraphVisualization()
    
    # Build the graph from CSV rows.
    for _, row in data.iterrows():
        node = row[0]
        neighbors = row[1:].dropna().tolist()
        graph[node] = neighbors
        for neighbor in neighbors:
            gv.add_edge(node, neighbor)
    
    start_node = input(f"Enter the starting node (available nodes: {list(graph.keys())}): ")
    if start_node not in graph:
        print("Invalid node! Exiting.")
        return
    
    print(f"Starting Node: {start_node}")
    
    print("Recursive BFS Traversal:")
    visited_recursive = set()
    recursive_tree_edges = []
    bfs_recursive(graph, [start_node], visited_recursive, recursive_tree_edges, {start_node})
    print("\n")
    
    print("Non-Recursive BFS Traversal:")
    tree_edges_non_recursive = bfs_non_recursive(graph, start_node)
    
    gv.visualize(gv.edges, recursive_tree_edges, tree_edges_non_recursive, start_node)

if __name__ == "__main__":
    main()
