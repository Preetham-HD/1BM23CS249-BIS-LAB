import numpy as np
import random

class ACO_Router:

    def __init__(self, cost_matrix, n_ants=10, n_iterations=100, alpha=1.0, beta=5.0, rho=0.5, Q=1.0):

        self.cost_matrix = np.array(cost_matrix)
        self.n_nodes = self.cost_matrix.shape[0]


        self.n_ants = n_ants # no of ants
        self.n_iterations = n_iterations #no of cycles
        self.alpha = alpha # Pheromone
        self.beta = beta # Heuristic
        self.rho = rho # Pheromone evaporation rate
        self.Q = Q # Pheromone deposit constant


        self.pheromone = np.ones((self.n_nodes, self.n_nodes)) / self.n_nodes


        self.best_path = None
        self.best_cost = np.inf

    def _get_heuristic(self, i, j):
        return 1.0 / (self.cost_matrix[i, j] + 1e-9)

    def _choose_next_node(self, current_node, visited):
        unvisited = [j for j in range(self.n_nodes) if j not in visited and self.cost_matrix[current_node, j] > 0]

        if not unvisited:
            return None


        probabilities = []
        for j in unvisited:
            tau = self.pheromone[current_node, j] # Pheromone
            eta = self._get_heuristic(current_node, j) # Heuristic


            probabilities.append((tau ** self.alpha) * (eta ** self.beta))


        total_prob = sum(probabilities)
        if total_prob == 0:

            return random.choice(unvisited)

        normalized_probs = [p / total_prob for p in probabilities]


        next_node = random.choices(unvisited, weights=normalized_probs, k=1)[0]
        return next_node

    def _run_ant(self, start_node, end_node):

        path = [start_node]
        current_node = start_node
        path_cost = 0

        while current_node != end_node:
            next_node = self._choose_next_node(current_node, path)

            if next_node is None:

                return None, np.inf

            path_cost += self.cost_matrix[current_node, next_node]
            path.append(next_node)
            current_node = next_node


            if len(path) > self.n_nodes:
                 return None, np.inf

        return path, path_cost

    def _update_pheromone(self, ant_paths):

        self.pheromone *= (1 - self.rho)


        for path, cost in ant_paths:
            if path is not None and cost < np.inf:
                deposit = self.Q / cost
                for i in range(len(path) - 1):
                    start, end = path[i], path[i+1]
                    self.pheromone[start, end] += deposit

    def find_shortest_path(self, start_node, end_node):
        for iteration in range(self.n_iterations):
            ant_paths = []

            for _ in range(self.n_ants):
                path, cost = self._run_ant(start_node, end_node)
                ant_paths.append((path, cost))

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path

            self._update_pheromone(ant_paths)

        return self.best_path, self.best_cost

COST_MATRIX = [
    [ 0, 7, 3, 0, 6, 11, 0, 0, 0, 0, 0, 0],
   [ 7, 0, 6, 0, 0, 7, 3, 0, 0, 0, 0, 0],
    [ 3, 6, 0, 12, 0, 0, 0, 9, 7, 0, 0, 0],
    [ 0, 0, 12, 0, 13, 0, 0, 0, 0, 0, 0, 0],
    [ 6, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 11],
    [11, 7, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0],
    [ 0, 3, 0, 0, 0, 0, 0, 0, 0, 11, 11, 0],
    [ 0, 0, 9, 0, 0, 0, 0, 0, 0, 18, 0, 0],
    [ 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 11, 18, 0, 0, 13, 0],
    [ 0, 0, 0, 0, 0, 12, 11, 0, 0, 13, 0, 0],
    [ 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0]
]

START_NODE = 0
END_NODE = 9

aco = ACO_Router(
    cost_matrix=COST_MATRIX,
    n_ants=20,
    n_iterations=50,
    alpha=1.0,
    beta=5.0,
    rho=0.1,
    Q=10.0
)

shortest_path, cost = aco.find_shortest_path(START_NODE, END_NODE)


node_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K'}
path_names = [node_map[n] for n in shortest_path] if shortest_path else "None"

print("--- ACO Router Traffic Management Simulation ---")
print(f"Start Node: {node_map[START_NODE]}, End Node: {node_map[END_NODE]}")
print(f"Iterations: {aco.n_iterations}, Ants per Iteration: {aco.n_ants}\n")

if shortest_path:
    print(f"Best Path Found (Node Indices): {shortest_path}")
    print(f"Best Path (Router Names): {' -> '.join(path_names)}")
    print(f"Total Cost/Delay: {cost}")
else:
    print("No path found.")


# Output:
# --- ACO Router Traffic Management Simulation ---
# Start Node: A, End Node: J
# Iterations: 50, Ants per Iteration: 20

# Best Path Found (Node Indices): [0, 2, 1, 6, 9]
# Best Path (Router Names): A -> C -> B -> G -> J
# Total Cost/Delay: 23
