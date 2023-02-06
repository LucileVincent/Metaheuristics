# Local search algorithm for travelling salesman problem
import random
import math
import numpy as np
import itertools


# Generate objective function
def f(x):
    # return np.sum(matrix[x[i], x[i + 1]] for i in range(len(x) - 1)) + matrix[x[-1], x[0]]
    return np.sum(np.fromiter((matrix[x[i], x[i + 1]] for i in range(len(x) - 1)), dtype=float)) + matrix[x[-1], x[0]]


# Generate a random solution
def generate_random_solution():
    x = np.random.permutation(len(matrix))
    return f(x), x


# Generate neighbours of a solution
def generate_neighbours(x):
    neighbours = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            neighbour = x.copy()
            neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
            neighbours.append(neighbour)
    return neighbours


# Find the best neighbour (the steepest ascent)
def find_best_neighbour_best(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    for neighbour in neighbours:
        if f(neighbour) < current_solution:
            best_neighbour = neighbour
            best_neighbour_solution = f(best_neighbour)
    return best_neighbour, best_neighbour_solution


# Find the best neighbour (First improvement)
def find_best_neighbour_first(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    for neighbour in neighbours:
        if f(neighbour) < current_solution:
            best_neighbour = neighbour
            best_neighbour_solution = f(best_neighbour)
            break
    return best_neighbour, best_neighbour_solution


# Find the best neighbour (Random improvement)
def find_best_neighbour_random(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    improved_neighbours = []
    for neighbour in neighbours:
        if f(neighbour) < current_solution:
            improved_neighbours.append(neighbour)
    if len(improved_neighbours) > 0:
        best_neighbour = random.choice(improved_neighbours)
        best_neighbour_solution = f(best_neighbour)
    return best_neighbour, best_neighbour_solution


# Local search algorithm
def local_search(matrix):
    best_value, best_perm = generate_random_solution()
    print("Initial solution : x =", best_perm, "s =", best_value)
    iteration = 1
    max_iteration = 1000
    while iteration < max_iteration:
        # Generate neighbours
        neighbors = generate_neighbours(best_perm)
        # Find the best neighbour
        current_perm, current_value = find_best_neighbour_best(neighbors, best_value)
        if current_value < best_value:
            best_value = current_value
            best_perm = current_perm
            print("Iteration", iteration, ": x =", best_perm, "s =", best_value)
        else:
            break
        iteration += 1
    return current_value, current_perm


if __name__ == "__main__":
    matrix = np.loadtxt('TSPDataset\\br17.39.atsp')
    iterate = 0
    max_iteration = 500
    nb_solution = 0
    best_value, best_perm = generate_random_solution()
    while iterate < max_iteration:
        current_value, current_perm = local_search(matrix)
        if current_value < best_value:
            best_value = current_value
            best_perm = current_perm
        if current_value == 39:
            nb_solution += 1
        iterate += 1
    print("Final solution : x =", best_perm, "s =", best_value, "nb solution =", nb_solution)