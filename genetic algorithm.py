# import random
# import math

# # Calculate Euclidean distance between two points in 3D space
# def euclidean_distance_3d(point1, point2):
#     return math.sqrt((point2[0] - point1[0]) ** 2 +
#                      (point2[1] - point1[1]) ** 2 +
#                      (point2[2] - point1[2]) ** 2)

# # Total distance of a path
# def calculate_path_distance(path, cities):
#     distance = 0
#     for i in range(len(path) - 1):
#         distance += euclidean_distance_3d(cities[path[i]], cities[path[i + 1]])
#     return distance

# # Generate initial population
# def generate_population(cities, population_size):
#     population = []
#     for _ in range(population_size):
#         individual = list(range(len(cities)))
#         random.shuffle(individual)
#         population.append(individual)
#     return population

# # Select parents (tournament selection)
# def select_parents(population, cities):
#     tournament_size = 5
#     tournament = random.sample(population, tournament_size)
#     best_individual = min(tournament, key=lambda path: calculate_path_distance(path, cities))
#     return best_individual

# # Crossover (ordered crossover)
# def crossover(parent1, parent2):
#     size = len(parent1)
#     start, end = sorted(random.sample(range(size), 2))
#     child = [-1] * size
#     child[start:end] = parent1[start:end]
    
#     pointer = 0
#     for gene in parent2:
#         if gene not in child:
#             while child[pointer] != -1:
#                 pointer += 1
#             child[pointer] = gene
#     return child

# # Mutate (swap mutation)
# def mutate(individual, mutation_rate):
#     for _ in range(len(individual)):
#         if random.random() < mutation_rate:
#             i, j = random.sample(range(len(individual)), 2)
#             individual[i], individual[j] = individual[j], individual[i]

# # Genetic Algorithm
# def genetic_algorithm(cities, population_size=100, generations=500, mutation_rate=0.02):
#     # Generate initial population
#     population = generate_population(cities, population_size)
    
#     # Evolve population over generations
#     for _ in range(generations):
#         new_population = []
#         for _ in range(population_size):
#             parent1 = select_parents(population, cities)
#             parent2 = select_parents(population, cities)
#             child = crossover(parent1, parent2)
#             mutate(child, mutation_rate)
#             new_population.append(child)
#         population = new_population
    
#     # Return the best solution
#     best_individual = min(population, key=lambda path: calculate_path_distance(path, cities))
#     return best_individual, calculate_path_distance(best_individual + [best_individual[0]], cities)

# # Parse input file
# def parse_input(file_name):
#     with open(file_name, 'r') as file:
#         lines = file.readlines()
#     n = int(lines[0].strip())  # Number of cities
#     cities = [tuple(map(int, line.strip().split())) for line in lines[1:n + 1]]
#     return n, cities

# # Write output to file
# def write_output(file_name, min_distance, path, cities):
#     with open(file_name, 'w') as file:
#         file.write(f"{min_distance:.2f}\n")
#         for city_index in path:
#             file.write(f"{' '.join(map(str, cities[city_index]))}\n")
#         # Add starting city at the end to complete the path
#         file.write(f"{' '.join(map(str, cities[path[0]]))}\n")

# # Main function
# def main():
#     input_file = "input.txt"
#     output_file = "output.txt"
    
#     # Parse input
#     _, cities = parse_input(input_file)
    
#     # Solve TSP using Genetic Algorithm
#     best_path, min_distance = genetic_algorithm(cities)
    
#     # Write output
#     write_output(output_file, min_distance, best_path, cities)

# # Run the program
# if __name__ == "__main__":
#     main()

import random
import math


# Compute Euclidean distance between two 3D points
def compute_distance(city1, city2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(city1, city2)))


# Read input from file and store locations
def load_cities(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    num_cities = int(lines[0].strip())
    locations = [tuple(map(int, line.strip().split())) for line in lines[1:]]
    return num_cities, locations


# Generate an initial population of random routes
def initialize_population(pop_size, num_cities):
    population = []
    for _ in range(pop_size):
        route = list(range(1, num_cities))  # Exclude the start city
        random.shuffle(route)
        route.insert(0, 0)  # Start at city 0
        route.append(0)  # End at city 0
        population.append(route)
    return population


# Evaluate fitness (inverse of total path distance)
def evaluate_fitness(population, cities):
    fitness_scores = []
    best_index = 0
    best_score = float("inf")

    for idx, route in enumerate(population):
        total_distance = sum(compute_distance(cities[route[i]], cities[route[i + 1]]) for i in range(len(route) - 1))
        fitness = 1 / total_distance  # Inverse distance for fitness

        fitness_scores.append(fitness)
        if total_distance < best_score:
            best_score = total_distance
            best_index = idx

    return fitness_scores, best_index


# Roulette wheel selection
def select_parent(fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [score / total_fitness for score in fitness_scores]
    
    r = random.random()
    cumulative_prob = 0
    for i, prob in enumerate(selection_probs):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return i
    return len(fitness_scores) - 1


# Crossover: Two-Point Order Crossover
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(1, size - 1), 2))  # Exclude start and end city

    child = [-1] * size
    child[start:end] = parent1[start:end]

    # Fill remaining cities from parent2
    pointer = 1  # Skip the start city (index 0)
    for city in parent2:
        if city not in child:
            while child[pointer] != -1:
                pointer += 1
            child[pointer] = city

    return child


# Swap mutation to introduce variation
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(route) - 1), 2)  # Avoid start/end cities
        route[i], route[j] = route[j], route[i]


# Genetic Algorithm main loop
def genetic_algorithm(cities, population_size=100, generations=100, mutation_rate=0.02):
    num_cities = len(cities)
    population = initialize_population(population_size, num_cities)

    for _ in range(generations):
        fitness_scores, best_index = evaluate_fitness(population, cities)
        new_population = []

        for _ in range(population_size):
            parent1 = population[select_parent(fitness_scores)]
            parent2 = population[select_parent(fitness_scores)]
            child = ordered_crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = sorted(new_population, key=lambda x: sum(compute_distance(cities[x[i]], cities[x[i + 1]]) for i in range(len(x) - 1)))[:population_size]

    best_path = population[0]
    best_distance = sum(compute_distance(cities[best_path[i]], cities[best_path[i + 1]]) for i in range(len(best_path) - 1))
    
    return best_distance, best_path


# Save the best solution to output.txt
def save_solution(filename, distance, path, cities):
    with open(filename, "w") as file:
        file.write(f"{distance:.3f}\n")
        for idx in path:
            file.write(" ".join(map(str, cities[idx])) + "\n")


# Main function
def main():
    input_file = "input.txt"
    output_file = "output.txt"

    num_cities, cities = load_cities(input_file)
    best_distance, best_route = genetic_algorithm(cities)

    save_solution(output_file, best_distance, best_route, cities)


if __name__ == "__main__":
    main()
