# Tuba Search
import random
import math
import matplotlib.pyplot as plt
import numpy as np

tsp_data = np.loadtxt('p01.15.291.tsp')


def pickneighbor(s):
    index = np.random.choice(range(15), 2, replace=False) # pick randomly 2 indices in [0,16]
    s[index[0]],s[index[1]] = s[index[1]],s[index[0]] # swap values located at these indices
    return s


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


def short_term_memory(s):

    return

def medium_term_memory(s):
    return s

def long_term_memory(s):
    return s


tabu_list = []
# random initial solution and its cost
s = np.random.choice(range(15), 15, replace=False)  # cities in [0,14]
cost_s = objectivefunction(s)

# save history of neighbor costs
tabu_list.append(cost_s)

# consider solution s as the best solution for now
best_s = np.copy(s)  # hard copy
best_cost = cost_s

while len(tabu_list) < 100:
    # pick a random neighbor and calculate its cost
    s_neighbor = pickneighbor(np.copy(s))
    cost_neighbor = objectivefunction(s_neighbor)

    # calculate difference of cost
    delta = cost_neighbor - cost_s

    # if cost of neighbor is better, accept it
    if delta < 0:
        s = s_neighbor
        cost_s = cost_neighbor

        # if cost of neighbor is better than best solution, update best solution
        if cost_neighbor < best_cost:
            best_s = s_neighbor
            best_cost = cost_neighbor

    # if cost of neighbor is worse, accept it with probability
    else:
        if accept_solution(delta, 1):
            s = s_neighbor
            cost_s = cost_neighbor

    # save history of neighbor costs
    tabu_list.append(cost_neighbor)

print('best sol, cost {} {}'.format(best_s,best_cost))
plt.plot(tabu_list)
plt.show()