import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math
from collections import deque

def compute_levels(edges, start):
    """Compute the level (distance from start) for each node using BFS."""
    tree = nx.DiGraph()
    tree.add_edges_from(edges)
    levels = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for child in tree.successors(node):
            if child not in levels:
                levels[child] = levels[node] + 1
                queue.append(child)
    return levels

def compute_positions(levels):
    """Assign positions to nodes so nodes on the same level are evenly spaced."""
    pos = {}
    level_nodes = {}
    for node, lvl in levels.items():
        level_nodes.setdefault(lvl, []).append(node)
    for lvl, nodes in level_nodes.items():
        count = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = 0.5 if count == 1 else (i + 1) / (count + 1)
            pos[node] = (x, -lvl)
    return pos

def filter_search_tree_edges(edges, destination):
    """Keep only the edges that eventually lead to the destination."""
    tree = nx.DiGraph()
    tree.add_edges_from([(u, v) for u, v, _ in edges])
    # Find all nodes that can reach the destination.
    reversed_tree = tree.reverse(copy=True)
    reachable = set(nx.descendants(reversed_tree, destination))
    reachable.add(destination)
    return [(u, v, attr) for u, v, attr in edges if u in reachable and v in reachable]

class GraphVisualization:
    def visualize(self, orig_edges, search_edges=None, start=None):
        """Visualize the original graph and, if provided, the filtered search tree."""
        # Build the original graph.
        orig_graph = nx.DiGraph()
        orig_graph.add_weighted_edges_from(orig_edges)
        
        if search_edges is None:
            plt.figure(figsize=(8, 7))
            plt.title("Original Directed Weighted Graph")
            pos = nx.spring_layout(orig_graph)
            nx.draw(orig_graph, pos, with_labels=True, node_color='lightblue',
                    node_size=500, edge_color='gray', arrows=True)
            labels = nx.get_edge_attributes(orig_graph, 'weight')
            nx.draw_networkx_edge_labels(orig_graph, pos, edge_labels=labels)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            # Left: Original graph.
            axes[0].set_title("Original Directed Weighted Graph")
            pos_orig = nx.spring_layout(orig_graph)
            nx.draw(orig_graph, pos=pos_orig, ax=axes[0], with_labels=True,
                    node_color='lightblue', node_size=500, edge_color='gray', arrows=True)
            labels = nx.get_edge_attributes(orig_graph, 'weight')
            nx.draw_networkx_edge_labels(orig_graph, pos_orig, edge_labels=labels, ax=axes[0])
            
            # Right: Filtered Best‑First Search Tree.
            tree = nx.DiGraph()
            tree.add_edges_from(search_edges)
            axes[1].set_title("Best‑First Search Tree (Filtered)")
            if start:
                levels = compute_levels(search_edges, start)
                pos_tree = compute_positions(levels)
            else:
                pos_tree = nx.spring_layout(tree)
            nx.draw(tree, pos_tree, ax=axes[1], with_labels=True, node_color='lightgreen',
                    node_size=500, edge_color='black', arrows=True)
            tree_labels = {(u, v): data.get("weight", "") for u, v, data in search_edges}
            nx.draw_networkx_edge_labels(tree, pos_tree, edge_labels=tree_labels, ax=axes[1])
        
        plt.tight_layout()
        plt.show()

def best_first_search(graph, source, destination, heuristic):
    """
    Perform Greedy Best‑First Search.
    Returns a tuple: (path as list of nodes, list of search tree edges).
    """
    open_list = []
    heapq.heappush(open_list, (heuristic.get(source, math.inf), source, [source]))
    visited = set()
    tree_edges = []
    
    while open_list:
        h_val, current, path = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)
        if current == destination:
            return path, tree_edges
        for neighbor, weight in graph.get(current, []):
            if neighbor not in visited:
                tree_edges.append((current, neighbor, {"weight": weight}))
                heapq.heappush(open_list, (heuristic.get(neighbor, math.inf), neighbor, path + [neighbor]))
    
    return None, tree_edges

def main():
    # Read the graph from CSV.
    file_path = "bestfs.csv"  # Make sure this CSV file is in your working directory.
    data = pd.read_csv(file_path)
    
    graph = {}
    orig_edges = []
    for _, row in data.iterrows():
        src = row["Source"]
        dst = row["Destination"]
        weight = float(row["Weight"])
        graph.setdefault(src, []).append((dst, weight))
        orig_edges.append((src, dst, weight))
    
    # Get all nodes from the CSV.
    all_nodes = set(data["Source"]).union(set(data["Destination"]))
    print("Available nodes:", sorted(all_nodes))
    
    # Interactive input.
    source = input("Enter the source node: ").strip()
    destination = input("Enter the destination node: ").strip()
    if source not in all_nodes or destination not in all_nodes:
        print("Source or destination node not found!")
        return
    
    # Define a non‑admissible heuristic if destination is "G", otherwise use zero.
    if destination == "G":
        heuristic = {
            "S": 5, "A": 1, "B": 10, "C": 8, "D": 7,
            "E": 6, "F": 9, "G": 0, "H": 4, "I": 3
        }
    else:
        heuristic = {node: 0 for node in all_nodes}
    
    print(f"\nSearching from {source} to {destination}...")
    path, tree_edges = best_first_search(graph, source, destination, heuristic)
    
    if path:
        total_cost = 0
        for i in range(len(path) - 1):
            for nbr, wt in graph[path[i]]:
                if nbr == path[i + 1]:
                    total_cost += wt
                    break
        print("Path found:", " -> ".join(path))
        print("Total cost:", total_cost)
        filtered_edges = filter_search_tree_edges(tree_edges, destination)
    else:
        print("No path found.")
        filtered_edges = None
    
    # Visualize the graph and the search tree.
    gv = GraphVisualization()
    gv.visualize(orig_edges, filtered_edges, start=source)

if __name__ == "__main__":
    main()
