import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class GraphVisualization:
    def __init__(self):
        self.edges = []

    def add_edge(self, a, b): 
        self.edges.append((a, b)) 

    def visualize(self, original_edges, traversal_tree_recursive=None, traversal_tree_non_recursive=None):
        original_graph = nx.Graph()
        original_graph.add_edges_from(original_edges)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        
        axes[0].set_title("Original Graph")
        nx.draw(original_graph, ax=axes[0], with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
        
        if traversal_tree_recursive:
            traversal_graph_recursive = nx.DiGraph()
            traversal_graph_recursive.add_edges_from(traversal_tree_recursive)
            axes[1].set_title("Recursive DFS Traversal")
            nx.draw(traversal_graph_recursive, ax=axes[1], with_labels=True, node_color='lightgreen', node_size=500, edge_color='black')
        
        if traversal_tree_non_recursive:
            traversal_graph_non_recursive = nx.DiGraph()
            traversal_graph_non_recursive.add_edges_from(traversal_tree_non_recursive)
            axes[2].set_title("Non-Recursive DFS Traversal")
            nx.draw(traversal_graph_non_recursive, ax=axes[2], with_labels=True, node_color='lightcoral', node_size=500, edge_color='black')

        plt.tight_layout()
        plt.show()

def dfs_recursive(graph, node, visited, tree_edges):
    visited.add(node)
    print(node, end=" ")
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            tree_edges.append((node, neighbor))  
            dfs_recursive(graph, neighbor, visited, tree_edges)
            

def dfs_non_recursive(graph, start_node):
    visited = set()
    stack = [(start_node, None)]  
    tree_edges = []

    while stack:
        node, parent = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
            if parent is not None:
                tree_edges.append((parent, node)) 

            for neighbor in graph.get(node, []): 
                if neighbor not in visited:
                    stack.append((neighbor, node)) 
    print()
    return tree_edges




def main():
    file_path = "/home/adyasha/Desktop/AI LAB/dfs.csv"
    data = pd.read_csv(file_path)

    graph = {} 
    gv = GraphVisualization()

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

    print("Recursive DFS Traversal:")
    visited_recursive = set()
    recursive_tree_edges = []
    dfs_recursive(graph, start_node, visited_recursive, recursive_tree_edges)
    print("\n")

    print("Non-Recursive DFS Traversal:")
    tree_edges_non_recursive = dfs_non_recursive(graph, start_node)

    gv.visualize(gv.edges, recursive_tree_edges, tree_edges_non_recursive)

if __name__ == "__main__":
    main()
