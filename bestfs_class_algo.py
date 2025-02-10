import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque

def best_first_search(graph, start_node, goal_node):
    """
    Best-First Search (Greedy Search)
    - Expands the least-cost node first.
    - Uses a priority queue to store nodes based on heuristic cost.
    - Tracks and returns the final path from start to goal.
    """
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start_node))  # Enqueue start node with cost 0
    parent = {start_node: None}  # Dictionary to reconstruct path
    tree_edges = []  # Stores traversal tree edges

    while not pq.empty():
        _, node = pq.get()  # Dequeue node with lowest cost

        if node in visited:
            continue

        visited.add(node)
        print(node, end=" ")

        if node == goal_node:
            break

        for neighbor, weight in sorted(graph.get(node, []), key=lambda x: x[1]):
            if neighbor not in visited:
                tree_edges.append((node, neighbor, {"weight": weight}))  # Add edge to tree
                pq.put((weight, neighbor))  # Enqueue neighbor with cost
                parent[neighbor] = node  # Track parent for path reconstruction

    print("\n")
    
    # Reconstruct the final path from start to goal
    path = []
    current = goal_node
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()  # Reverse the path to get start → goal order
    
    print(f"Final Path Traversed: {' → '.join(path)}")
    
    return tree_edges, path  # Return tree structure and final path

class GraphVisualization:
    def __init__(self):
        self.edges = []

    def add_edge(self, a, b, weight):
        self.edges.append((a, b, weight))

    def visualize(self, original_edges, traversal_tree, path):
        """Visualizes the original directed graph and BFS traversal tree"""
        original_graph = nx.DiGraph()  # Now using a Directed Graph
        original_graph.add_weighted_edges_from(original_edges)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Plot directed original graph
        axes[0].set_title("Original Directed Graph")
        pos_orig = nx.spring_layout(original_graph)
        nx.draw(original_graph, pos=pos_orig, ax=axes[0], with_labels=True,
                node_color='lightblue', node_size=500, edge_color='gray', arrows=True)
        edge_labels = nx.get_edge_attributes(original_graph, 'weight')
        nx.draw_networkx_edge_labels(original_graph, pos_orig, edge_labels=edge_labels, ax=axes[0])

        # Plot Best-First Search Traversal Tree
        if traversal_tree:
            bfs_graph = nx.DiGraph()
            bfs_graph.add_edges_from([(u, v) for u, v, _ in traversal_tree])

            axes[1].set_title("Best-First Search Traversal Tree")
            pos = self.compute_positions(traversal_tree, path[0])
            nx.draw(bfs_graph, pos=pos, ax=axes[1], with_labels=True, node_color='lightgreen',
                    node_size=500, edge_color='black', arrows=True)

            edge_labels_tree = {(u, v): d["weight"] for u, v, d in traversal_tree}
            nx.draw_networkx_edge_labels(bfs_graph, pos, edge_labels=edge_labels_tree, ax=axes[1])

        plt.tight_layout()
        plt.show()

    def compute_positions(self, tree_edges, start_node):
        """Computes hierarchical positions for the BFS tree"""
        levels = {start_node: 0}
        queue = deque([start_node])

        for u, v, _ in tree_edges:
            if v not in levels:
                levels[v] = levels[u] + 1
                queue.append(v)

        level_nodes = {}
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)

        pos = {}
        for lvl, nodes in level_nodes.items():
            for i, node in enumerate(sorted(nodes)):
                x = 0.5 if len(nodes) == 1 else (i + 1) / (len(nodes) + 1)
                pos[node] = (x, -lvl)

        return pos

def load_graph_from_csv(file_path):
    """
    Loads a weighted graph from a CSV file where:
    - Each row represents a node and its neighbors with weights.
    - Graph is **directed**, ensuring one-way relationships.
    """
    data = pd.read_csv(file_path)
    graph = {}
    original_edges = []

    for _, row in data.iterrows():
        node = row["Node"]
        if node not in graph:
            graph[node] = []

        for i in range(1, 5):
            neighbor_col = f"Neighbor{i}"
            weight_col = f"Weight{i}"
            neighbor = row.get(neighbor_col)
            weight = row.get(weight_col)

            if pd.notna(neighbor) and str(neighbor).strip() != "":
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    weight = 1.0  

                graph[node].append((neighbor, weight))

                # Store directed edges (one-way relationships)
                original_edges.append((node, neighbor, weight))

    return graph, original_edges

def main():
    file_path = "bfs.csv"  # Ensure the correct file path

    try:
        graph, original_edges = load_graph_from_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    available_nodes = list(graph.keys())

    start_node = input(f"Enter the starting node (available nodes: {available_nodes}): ").strip()
    if start_node not in graph:
        print("Invalid node! Exiting.")
        return

    goal_node = input(f"Enter the goal node (available nodes: {available_nodes}): ").strip()
    if goal_node not in graph:
        print("Invalid node! Exiting.")
        return

    print(f"\nStarting Node: {start_node}, Goal Node: {goal_node}")
    print("\nBest-First Search Traversal:")
    
    bfs_tree_edges, path = best_first_search(graph, start_node, goal_node)

    gv = GraphVisualization()
    gv.visualize(original_edges, bfs_tree_edges, path)

if __name__ == "__main__":
    main()
