# Tuba Search
import random
import math
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tsp_data = np.loadtxt('gr17.2085.tsp')


def pickneighbor(s):
    index = np.random.choice(range(17), 2, replace=False) # pick randomly 2 indices in [0,16]
    s[index[0]],s[index[1]] = s[index[1]],s[index[0]] # swap values located at these indices
    return s


def create_neighbours(path):
    permutations = []
    for i in range(17):
        for j in range(i + 1, 17):
            permutations.append((i, j))

    paths = []
    for permutation in permutations:
        path_copy = deepcopy(path)
        tmp = path_copy[permutation[0]]
        path_copy[permutation[0]] = path_copy[permutation[1]]
        path_copy[permutation[1]] = tmp
        paths.append(path_copy)
    return paths


def objectivefunction(s):
    cost = 0
    for i in range(s.shape[0]-1):
        cost = cost + tsp_data[s[i]][s[i+1]]

    cost = cost + tsp_data[s[-1]][s[0]]
    return cost


def accept_solution(delta, T):
    decision = ['yes','no']
    prob = math.exp(-abs(delta)/T)  # probability of acceptance
    final = np.random.choice(decision, 1, p=[prob, 1-prob])  # select yes or no according to p
    if final == 'yes':
        return True
    else:
        return False


def tabu_search(tabu_list):
    # random initial solution and its cost
    s = np.random.choice(range(17), 17, replace=False)  # cities in [0,14]
    cost_s = objectivefunction(s)

    # save history of neighbor costs
    tabu_list.append(cost_s)

    T = 1

    # consider solution s as the best solution for now
    best_s = np.copy(s)  # hard copy
    best_cost = cost_s

    while len(tabu_list) < 1000:
        # pick a random neighbor and calculate its cost
        s_neighbor = pickneighbor(np.copy(s))
        cost_neighbor = objectivefunction(s_neighbor)

        # calculate difference of cost
        delta = cost_neighbor - cost_s

        # accept solution if it is better
        if delta < 0:
            s = np.copy(s_neighbor)
            cost_s = cost_neighbor
            # if cost of neighbor is better than best solution, update best solution
            if cost_neighbor < best_cost:
                best_s = s_neighbor
                best_cost = cost_neighbor

        # accept solution if it is worse but with a certain probability
        elif accept_solution(delta, T):
            s = np.copy(s_neighbor)
            cost_s = cost_neighbor

        # save history of neighbor costs
        tabu_list.append(cost_s)

    return s, tabu_list, cost_s

def main():
    tabu_list = []
    s, tabu_list, best_s = tabu_search(tabu_list)
    print(s)
    print(tabu_list)
    print(best_s)
    plt.plot(tabu_list)
    plt.show()

main()