import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def bfs_using_queue(graph, start_node):
    """
    BFS using a queue:
    - Start at an arbitrary node.
    - Store it onto the queue (enqueue).
    - Repeat until the queue is empty:
      - Remove a node from the queue (dequeue).
      - If the node is not visited:
        - Mark it as visited.
        - Store all unvisited neighbors onto the queue.
    """
    visited = set()
    queue = deque([start_node])
    tree_edges = []
    
    while queue:
        node = queue.popleft()  # Dequeue
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
            for neighbor, weight in graph.get(node, []):
                if neighbor not in visited and neighbor not in queue:
                    tree_edges.append((node, neighbor, {"weight": weight}))
                    queue.append(neighbor)  # Enqueue unvisited neighbor
    print()
    return tree_edges

class GraphVisualization:
    def __init__(self):
        self.edges = []
    
    def add_edge(self, a, b, weight):
        self.edges.append((a, b, weight))
    
    def visualize(self, original_edges, traversal_tree, start_node=None):
        original_graph = nx.Graph()
        original_graph.add_weighted_edges_from(original_edges)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].set_title("Original Weighted Graph")
        pos_orig = nx.spring_layout(original_graph)
        nx.draw(original_graph, pos=pos_orig, ax=axes[0], with_labels=True,
                node_color='lightblue', node_size=500, edge_color='gray')
        edge_labels = nx.get_edge_attributes(original_graph, 'weight')
        nx.draw_networkx_edge_labels(original_graph, pos_orig, edge_labels=edge_labels, ax=axes[0])
        
        # BFS Tree Visualization
        if traversal_tree:
            bfs_graph = nx.DiGraph()
            bfs_graph.add_edges_from(traversal_tree)
            axes[1].set_title("BFS Traversal Tree")
            pos = self.compute_positions(traversal_tree, start_node)
            nx.draw(bfs_graph, pos=pos, ax=axes[1], with_labels=True, node_color='lightgreen', node_size=500, edge_color='black', arrows=True)
            edge_labels_tree = { (u, v): d["weight"] for u, v, d in traversal_tree }
            nx.draw_networkx_edge_labels(bfs_graph, pos, edge_labels=edge_labels_tree, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def compute_positions(self, tree_edges, start_node):
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

def main():
    file_path = "/home/adyasha/Desktop/AI LAB/bfs.csv"  
    data = pd.read_csv(file_path)
    graph = {}
    gv = GraphVisualization()
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
                if neighbor not in graph:
                    graph[neighbor] = []
                if (node, weight) not in graph[neighbor]:
                    graph[neighbor].append((node, weight))
                if str(node) < str(neighbor):
                    original_edges.append((node, neighbor, weight))
    
    available_nodes = list(graph.keys())
    start_node = input(f"Enter the starting node (available nodes: {available_nodes}): ").strip()
    if start_node not in graph:
        print("Invalid node! Exiting.")
        return
    
    print(f"Starting Node: {start_node}")
    print("\nBFS Traversal using Queue:")
    bfs_tree_edges = bfs_using_queue(graph, start_node)
    
    gv.visualize(original_edges, bfs_tree_edges, start_node)

if __name__ == "__main__":
    main()
