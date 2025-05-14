import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import ListedColormap

# Create a function to load the maze from a text file
def load_maze(file):
    maze = np.loadtxt(file, dtype=int)
    return maze


# Generar población con genomas aleatorios
def generate_population(size, genome_length):
    return [random.choices(DIRECTIONS, k=genome_length) for _ in range(size)]

# Direcciones posibles y movimientos
DIRECTIONS = ['U', 'D', 'L', 'R']
MOVES = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


# Calculates fitness based on proximity, exploration, and penalties.
def reward(individual, maze, start, end):
    x, y = start
    visited = set()
    steps = 0
    collisions = 0
    for move in individual:
        dx, dy = MOVES[move]
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            x, y = nx, ny
        else:
            collisions += 1
        visited.add((x, y))
        steps += 1
        if (x, y) == end:
            break
    dist_to_goal = abs(x - end[0]) + abs(y - end[1])
    reached_goal = (x, y) == end
    fitness = -dist_to_goal - 5 * collisions + (1000 if reached_goal else 0)
    return fitness

# Selects genomes for the next generation based on fitness.
def select(population, fitnesses, k=3):
    selected = []
    k = min(k, len(population))
    for _ in range(len(population)):
        contenders = random.sample(list(zip(population, fitnesses)), k)
        winner = max(contenders, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

# Combines genes from two parents to produce offspring.
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]


# Randomly mutates genes in the genome based on the mutation rate.
def mutate(individual, mutation_rate):
    return [gene if random.random() > mutation_rate else random.choice(DIRECTIONS) for gene in individual]

# Main loop that runs the GA for a specified number of generations.
def evolve(population, maze, start, end, generations=100, mutation_rate=0.1, elitism=0.02,crossover_rate=0.8):

    best = None
    best_fitness = float('-inf')
    best_over_time = []

    population_size = len(population)
    elite_count = max(1, int(population_size * elitism))

    for gen in range(generations):
        fitnesses = [reward(ind, maze, start, end) for ind in population]

        # Encontrar el mejor de esta generación
        current_best = max(zip(population, fitnesses), key=lambda x: x[1])
        if current_best[1] > best_fitness:
            best = current_best[0]
            best_fitness = current_best[1]

        best_over_time.append(best_fitness)

        # conservar los mejores individuos, elitismo
        elite_individuals = [ind for ind, _ in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_count]]

        # Selección y reproducción
        selected = select(population, fitnesses)
        offspring = []

        for i in range(0, len(selected) - elite_count, 2):
            p1 = selected[i]
            p2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            if random.random() < crossover_rate:
                child = crossover(p1, p2)
            else:
                child = p1[:]
            child = mutate(child, mutation_rate)
            offspring.append(child)

        # Nueva población = élite + descendencia
        population = elite_individuals + offspring[:population_size - elite_count]

    return best, best_fitness, best_over_time


def simulate_path(individual, maze, start, end):
    x, y = start
    path = [(x, y)]
    for move in individual:
        dx, dy = MOVES[move]
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            x, y = nx, ny
        path.append((x, y))
        if (x, y) == end:
            break
    return path

def display_maze(maze, path, start, end):
    maze_copy = maze.copy()

    # Marcar camino
    for (x, y) in path:
        if (x, y) != start and (x, y) != end:
            maze_copy[x][y] = 2  # ruta

    # Marcar entrada y salida
    maze_copy[start[0]][start[1]] = 3
    maze_copy[end[0]][end[1]] = 4

    cmap = ListedColormap(['white', 'black', 'blue', 'green', 'red'])

    plt.figure(figsize=(6, 6))
    plt.pcolor(maze_copy[::-1], cmap=cmap, edgecolors='k', linewidths=2)
    plt.gca().set_aspect('equal')
    plt.xticks([])
    plt.yticks([])
    plt.title('Mejor individuo en el laberinto')
    plt.show()


if __name__ == "__main__":
    maze = load_maze("maze_case_base.txt")
    start = (1, 0)
    end = (13, 14)
    pop = generate_population(100, 450)
    best, best_fit, fitness_history  = evolve(pop, maze, start, end, generations=100)
    print("Genoma del mejor individuo:", best)
    print("Fitness:", best_fit)
    path = simulate_path(best, maze, start, end)
    display_maze(maze, path, start, end)
    np.savetxt('maze.txt', maze, fmt='%d')