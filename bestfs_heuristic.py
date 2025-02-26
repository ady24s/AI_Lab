import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from collections import deque

def read_weighted_graph_from_csv(filename):
    
    df = pd.read_csv(filename)
    graph = {}
    for _, row in df.iterrows():
        u = str(row["Source"]).strip()
        v = str(row["Destination"]).strip()
        try:
            cost = float(row["Weight"])
        except (ValueError, TypeError):
            cost = 1.0
        if u not in graph:
            graph[u] = []
        graph[u].append((v, cost))
        # Ensure the destination appears even if it has no outgoing edges.
        if v not in graph:
            graph[v] = []
    return graph


def best_first_search(graph, heuristic, start, target):
   
    visited = set()
    pq = PriorityQueue()
    parent = {}
    parent[start] = None
    pq.put((heuristic[start], start, []))
    
    while not pq.empty():
        h_val, u, path = pq.get()
        path = path + [u]
        if u == target:
            return path, parent
        if u not in visited:
            visited.add(u)
            for v, cost in graph.get(u, []):
                if v not in visited and v not in parent:
                    parent[v] = u
                    pq.put((heuristic[v], v, path))
    return [], parent

class GraphVisualization:
    def compute_positions(self, edges, start_node):
        
        tempG = nx.DiGraph()
        for e in edges:
            if len(e) == 2:
                u, v = e
            else:
                u, v, _ = e
            tempG.add_edge(u, v)
        # If start_node is not in tempG, add it to avoid KeyError.
        if start_node not in tempG:
            tempG.add_node(start_node)
        # Compute BFS levels starting from start_node.
        levels = {start_node: 0}
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            for neighbor in tempG.successors(node):
                if neighbor not in levels:
                    levels[neighbor] = levels[node] + 1
                    queue.append(neighbor)
        # Group nodes by level.
        level_nodes = {}
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
        pos = {}
        for lvl, nodes in level_nodes.items():
            nodes_sorted = sorted(nodes)
            for i, node in enumerate(nodes_sorted):
                x = 0.5 if len(nodes_sorted) == 1 else (i + 1) / (len(nodes_sorted) + 1)
                pos[node] = (x, -lvl)
        return pos

    def compute_positions_for_path(self, path):
        
        pos = {}
        n = len(path)
        for i, node in enumerate(path):
            pos[node] = ((i+1)/(n+1), 0)
        return pos

    def visualize_all(self, original_edges, optimal_path, source, target):
        
        # Build the original graph.
        orig_graph = nx.DiGraph()
        orig_graph.add_weighted_edges_from(original_edges)
        
        all_paths = list(nx.all_simple_paths(orig_graph, source, target))
        union_edges = set()
        for path in all_paths:
            for u, v in zip(path, path[1:]):
                union_edges.add((u, v))
        nonopt_graph = nx.DiGraph()
        nonopt_graph.add_edges_from(union_edges)
        
        #non-optimal
        non_optimal_path = None
        for p in all_paths:
            if p != optimal_path:
                non_optimal_path = p
                break
        if non_optimal_path is None and all_paths:
            non_optimal_path = all_paths[0]
        
        # optimal
        opt_graph = nx.DiGraph()
        opt_edges = list(zip(optimal_path, optimal_path[1:]))
        opt_graph.add_edges_from(opt_edges)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].set_title("Original Weighted Directed Graph")
        pos_orig = nx.spring_layout(orig_graph)
        nx.draw(orig_graph, pos=pos_orig, ax=axes[0], with_labels=True,
                node_color='lightblue', node_size=500, edge_color='gray', arrows=True)
        edge_labels_orig = nx.get_edge_attributes(orig_graph, 'weight')
        nx.draw_networkx_edge_labels(orig_graph, pos_orig, edge_labels=edge_labels_orig, ax=axes[0])
        
        axes[1].set_title("Optimal Path (Least Cost)")
        pos_opt = self.compute_positions_for_path(optimal_path)
        nx.draw(opt_graph, pos=pos_opt, ax=axes[1], with_labels=True,
                node_color='lightgreen', node_size=500, edge_color='red', arrows=True)
        edge_labels_opt = {}
        for u, v in opt_edges:
            for edge in original_edges:
                if edge[0] == u and edge[1] == v:
                    edge_labels_opt[(u, v)] = edge[2]
                    break
        nx.draw_networkx_edge_labels(opt_graph, pos_opt, edge_labels=edge_labels_opt, ax=axes[1])
        
        axes[2].set_title("Non-Optimal Paths")
        pos_nonopt = self.compute_positions(list(nonopt_graph.edges()), source)
        nx.draw(nonopt_graph, pos=pos_nonopt, ax=axes[2], with_labels=True,
                node_color='salmon', node_size=500, edge_color='gray', arrows=True)
        edge_labels_nonopt = {}
        for u, v in nonopt_graph.edges():
            for edge in original_edges:
                if edge[0] == u and edge[1] == v:
                    edge_labels_nonopt[(u, v)] = edge[2]
                    break
        nx.draw_networkx_edge_labels(nonopt_graph, pos_nonopt, edge_labels=edge_labels_nonopt, ax=axes[2])
        if non_optimal_path and len(non_optimal_path) > 1:
            nonopt_path_edges = list(zip(non_optimal_path, non_optimal_path[1:]))
            nx.draw_networkx_edges(nonopt_graph, pos_nonopt, edgelist=nonopt_path_edges,
                                   edge_color='blue', width=2, arrows=True, ax=axes[2])
        
        plt.tight_layout()
        plt.show()

def main():
    graph_filename = "bestfs.csv" 
    graph = read_weighted_graph_from_csv(graph_filename)
    
    heuristic = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 3,
        'F': 0,   # Destination F
        'G': 2,
        'H': 3,
        'I': 2,
        'J': 4,
        'K': 3,
        'L': 2,
        'M': 2,
        'N': 1,
        'O': 2
    }
    
    print("Nodes in the graph:", list(graph.keys()))
    source = input("Enter source node: ").strip()
    target = "F"  
    print("Destination node is hardcoded as 'F'")
    
    if source == target:
        print("Source is the same as target. Optimal path is trivial.")
        return
    
    if source not in graph or target not in heuristic:
        print("Invalid node(s)! Check your input.")
        return
    
    optimal_path, parent = best_first_search(graph, heuristic, source, target)
    if optimal_path:
        print("Optimal (Least-Cost) Path:", " -> ".join(optimal_path))
    else:
        print("No path found.")
        return
    
    original_edges = []
    for u in graph:
        for v, cost in graph[u]:
            original_edges.append((u, v, cost))
    
    gv = GraphVisualization()
    gv.visualize_all(original_edges, optimal_path, source, target)

if __name__ == "__main__":
    main()
