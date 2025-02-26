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
        if v not in graph:
            graph[v] = []
    return graph

def a_star_search(graph, heuristic, start, target):
    visited = set()
    pq = PriorityQueue()
    parent = {}
    g_costs = {node: float('inf') for node in graph}
    g_costs[start] = 0
    pq.put((heuristic[start], start, []))
    
    while not pq.empty():
        f_val, u, path = pq.get()
        path = path + [u]
        if u == target:
            return path, parent
        if u not in visited:
            visited.add(u)
            for v, cost in graph.get(u, []):
                new_g_cost = g_costs[u] + cost
                if new_g_cost < g_costs[v]:
                    g_costs[v] = new_g_cost
                    parent[v] = u
                    pq.put((new_g_cost + heuristic[v], v, path))
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
        if start_node not in tempG:
            tempG.add_node(start_node)
        levels = {start_node: 0}
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            for neighbor in tempG.successors(node):
                if neighbor not in levels:
                    levels[neighbor] = levels[node] + 1
                    queue.append(neighbor)
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
        orig_graph = nx.DiGraph()
        orig_graph.add_weighted_edges_from(original_edges)
        all_paths = list(nx.all_simple_paths(orig_graph, source, target))
        union_edges = set()
        for path in all_paths:
            for u, v in zip(path, path[1:]):
                union_edges.add((u, v))
        nonopt_graph = nx.DiGraph()
        nonopt_graph.add_edges_from(union_edges)
        non_optimal_path = None
        for p in all_paths:
            if p != optimal_path:
                non_optimal_path = p
                break
        if non_optimal_path is None and all_paths:
            non_optimal_path = all_paths[0]
        opt_graph = nx.DiGraph()
        opt_edges = list(zip(optimal_path, optimal_path[1:]))
        opt_graph.add_edges_from(opt_edges)

        # Figure 1: Optimal Path
        fig1 = plt.figure(figsize=(8, 6))  
        plt.title("Optimal Path")
        pos_orig = nx.spring_layout(orig_graph)
        nx.draw(orig_graph, pos=pos_orig, with_labels=True,
                node_color='lightblue', node_size=500, edge_color='gray', arrows=True)
        
        # Draw the optimal path over the original graph
        if optimal_path:
            opt_path_edges = list(zip(optimal_path, optimal_path[1:]))
            nx.draw_networkx_edges(orig_graph, pos=pos_orig, edgelist=opt_path_edges,
                                   edge_color='red', width=2, arrows=True)

        # Add edge labels for the optimal path
        edge_labels_orig = nx.get_edge_attributes(orig_graph, 'weight')
        nx.draw_networkx_edge_labels(orig_graph, pos_orig, edge_labels=edge_labels_orig)

        # Figure 2: Non-Optimal Path
        fig2 = plt.figure(figsize=(8, 6))
        plt.title("Non-Optimal Path")
        pos_nonopt = self.compute_positions(list(nonopt_graph.edges()), source)
        nx.draw(orig_graph, pos=pos_nonopt, with_labels=True, 
                node_color='lightblue', node_size=500, edge_color='gray', arrows=True)
        
        # Draw the non-optimal path over the original graph
        if non_optimal_path and len(non_optimal_path) > 1:
            nonopt_path_edges = list(zip(non_optimal_path, non_optimal_path[1:]))
            nx.draw_networkx_edges(orig_graph, pos=pos_nonopt, edgelist=nonopt_path_edges,
                                   edge_color='blue', width=2, arrows=True)

        # Add edge labels for the non-optimal path
        edge_labels_nonopt = {}
        for u, v in nonopt_graph.edges():
            for edge in original_edges:
                if edge[0] == u and edge[1] == v:
                    edge_labels_nonopt[(u, v)] = edge[2]
                    break
        nx.draw_networkx_edge_labels(orig_graph, pos_nonopt, edge_labels=edge_labels_nonopt)

        plt.tight_layout()
        plt.show()

def main():
    graph_filename = "bestfs.csv" 
    graph = read_weighted_graph_from_csv(graph_filename)
    heuristic = {
        'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 0, 'G': 2, 'H': 3, 'I': 2,
        'J': 4, 'K': 3, 'L': 2, 'M': 2, 'N': 1, 'O': 2
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
    optimal_path, parent = a_star_search(graph, heuristic, source, target)
    if optimal_path:
        print("Optimal (Least-Cost) Path:", " -> ".join(optimal_path))
    else:
        print("No path found.")
        return
    original_edges = [(u, v, cost) for u in graph for v, cost in graph[u]]
    gv = GraphVisualization()
    gv.visualize_all(original_edges, optimal_path, source, target)

if __name__ == "__main__":
    main()
