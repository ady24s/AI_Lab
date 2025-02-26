import pandas as pd
import heapq
import networkx as nx
import matplotlib.pyplot as plt

class BestFirstSearch:
    def __init__(self, csv_file, goal):
        self.graph = {}  # Adjacency list representation
        self.goal = goal
        self.load_graph(csv_file)
        self.heuristics = self.define_heuristics()  # Hardcoded heuristic function
    
    def load_graph(self, csv_file):
        """ Load graph edges from a CSV file. """
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            src, dest, cost = row['Source'], row['Destination'], row['Weight']
            
            # Add edges to the graph
            if src not in self.graph:
                self.graph[src] = []
            self.graph[src].append((dest, cost))

            if dest not in self.graph:
                self.graph[dest] = []

    def define_heuristics(self):
        """ Hardcoded heuristic values for nodes. """
        return {
            'A': 7,
            'B': 6,
            'C': 2,
            'D': 5,
            'E': 5,
            'F': 0,  # Goal node (F) heuristic is 0
            'G': 3,
            'H': 6,
            'I': 7,
            'J': 8,
            'K': 7,
            'L': 4,
            'M': 3,
            'N': 5,
            'O': 6,
            'P': 2,  # Additional Nodes
            'Q': 4,
            'R': 6,
            'S': 3,
            'T': 5
        }
    
    def visualize_graph(self):
        """ Display the original graph using NetworkX. """
        G = nx.DiGraph()
        
        for node, neighbors in self.graph.items():
            for neighbor, weight in neighbors:
                G.add_edge(node, neighbor, weight=weight)
        
        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, edge_color="gray")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Original Graph Representation")
        plt.show()

    def best_first_search(self, start):
        """ Perform Best-First Search from start node to goal node. """
        priority_queue = []
        heapq.heappush(priority_queue, (self.heuristics[start], start))  # (heuristic, node)
        visited = set()
        path = []
        non_optimal_traversal = []  # Stores all visited nodes

        while priority_queue:
            heuristic, node = heapq.heappop(priority_queue)
            if node in visited:
                continue
            
            path.append(node)
            visited.add(node)
            non_optimal_traversal.append(node)

            if node == self.goal:
                return path, non_optimal_traversal  # Return optimal path and visited nodes

            for neighbor, _ in self.graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (self.heuristics.get(neighbor, float('inf')), neighbor))

        return None, non_optimal_traversal  # No path found

# Example Usage
csv_file = "bestfs.csv"  # Ensure this CSV exists in the same directory
start_node = "A"
goal_node = "F"

search = BestFirstSearch(csv_file, goal_node)

# Display the original graph
search.visualize_graph()

# Perform Best-First Search
optimal_path, visited_nodes = search.best_first_search(start_node)

if optimal_path:
    print("\nOptimal Best-First Search Path:", " -> ".join(optimal_path))
else:
    print("\nNo optimal path found.")

print("\nNon-Optimal Traversal (All Visited Nodes):", " -> ".join(visited_nodes))
