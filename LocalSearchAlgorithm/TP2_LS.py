# Local search algorithm

import random
import numpy as np
import copy

# Generate a random solution between 0 and 31
def generate_random_solution():
    # x = np.random.randint(size=5, low=0, high=2)
    x = random.randint(0, 31)
    return maximizationfct(x), x


# Convert a binary array of length 5 to an int
def de_binarise(binary):
    return int("".join(str(i) for i in binary), 2)


# Convert an int to a binary array of length 5
def binarise(x):
    return [int(i) for i in bin(x)[2:].zfill(5)]


# Objective function
def maximizationfct(s):
    # change binaire en nb
    # x = sum([e*2**(len(s)-i-1) for i,e in enumerate(s)])
    return s**3 - 60*s**2 + 900*s


# création des voisins
def array_neighbors(s):
    s = binarise(s)
    neighbors = []
    for i in range(len(s)):
        neighbor = s.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors


def solution(s):
    sol_ini = s.copy()
    neighbors = array_neighbors(sol_ini)
    solution = []
    for i in neighbors:
        solution.append(maximizationfct(i))
    return solution


# Find the best neighbour (the steepest ascent)
def find_best_neighbour_best(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    for neighbour in neighbours:
        if maximizationfct(de_binarise(neighbour)) > current_solution:
            best_neighbour = neighbour
            best_neighbour_solution = maximizationfct(de_binarise(best_neighbour))
        return best_neighbour, best_neighbour_solution


# Find the best neighbour (First improvement)
def find_best_neighbour_first(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    for neighbour in neighbours:
        if maximizationfct(de_binarise(neighbour)) > current_solution:
            best_neighbour = neighbour
            best_neighbour_solution = maximizationfct(de_binarise(best_neighbour))
            break
    return best_neighbour, best_neighbour_solution


# Find the best neighbour (Random improvement)
def find_best_neighbour_random(neighbours, current_solution):
    best_neighbour_solution = current_solution
    best_neighbour = neighbours[0]
    improved_neighbours = []
    for neighbour in neighbours:
        if maximizationfct(de_binarise(neighbour)) > current_solution:
            improved_neighbours.append(neighbour)
    if len(improved_neighbours) > 0:
        best_neighbour = random.choice(improved_neighbours)
        best_neighbour_solution = maximizationfct(de_binarise(best_neighbour))
    return best_neighbour, best_neighbour_solution


# Local search algorithm
def local_search():
    iteration = 1
    max_iteration = 1000
    # Generate a random solution
    current_solution, x = generate_random_solution()
    print("Initial solution : x =", x, "s =", current_solution)
    while iteration < max_iteration:
        # Generate neighbours
        neighbours = array_neighbors(x)
        # Find the best neighbour
        best_neighbour, best_neighbour_solution = find_best_neighbour_best(neighbours, current_solution)
        if best_neighbour_solution > current_solution:
            current_solution = best_neighbour_solution
            x = de_binarise(best_neighbour)
            print("Iteration", iteration, ": x =", x, "s =", current_solution)
        else:
            break
        iteration += 1
    return x, current_solution


if __name__ == "__main__":
    search_iterations = 1
    good_solutions = 0
    while search_iterations <= 10:
        print("Search iteration", search_iterations)
        x, current_solution = local_search()
        if current_solution == 4000:
            good_solutions += 1
        search_iterations += 1
    print("Final solution : x =", x, "s =", current_solution)
    print("Good solutions :", good_solutions)




