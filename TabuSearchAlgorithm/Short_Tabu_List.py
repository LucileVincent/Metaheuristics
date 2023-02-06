import queue
import math
import matplotlib.pyplot as plt
import numpy as np

tsp_data = np.loadtxt('gr17.2085.tsp')


def tabu_neighbor(s, tabu_list):
    for sol in tabu_list.queue:
        if ((sol == s).all()):
            return True

    return False


def pickneighbor(s):
    index = np.random.choice(range(17), 2, replace=False) # pick randomly 2 indices in [0,16]
    s[index[0]],s[index[1]] = s[index[1]],s[index[0]] # swap values located at these indices
    return s


def objectivefunction(s):
    cost = 0
    for i in range(s.shape[0]-1):
        cost = cost + tsp_data[s[i]][s[i+1]]

    cost = cost + tsp_data[s[-1]][s[0]]
    return cost


def best_admessible_neighbor(s, tabu_list):
    while True:
        neighbor_sol = pickneighbor(np.copy(s))
        tabu = tabu_neighbor(neighbor_sol, tabu_list)

        if (objectivefunction(neighbor_sol) < objectivefunction(s)):
            if tabu:
                return neighbor_sol, True
            else:
                return neighbor_sol, False
        else:
            if not (tabu):
                return neighbor_sol, False


tabu_list = queue.Queue(5)  # FIFO queue of 5 items max
costs_hist = []

# random initial solution and its cost
s = np.random.choice(range(17), 17, replace=False)  # cities in [0,16]
cost_s = objectivefunction(s)
# update tabu list
tabu_list.put(np.copy(s))
# save history of costs
costs_hist.append(cost_s)
# consider solution s as the best solution for now
best_s = np.copy(s)  # hard copy
best_cost = cost_s
total_it = 0  # max iterations
while total_it < 8000:
    adm_s, tabu = best_admessible_neighbor(np.copy(s), tabu_list)
    cost_adm_s = objectivefunction(adm_s)
    costs_hist.append(cost_adm_s)

    # update s
    s = np.copy(adm_s)

    if (not (tabu)):  # adm_s not satisfying an aspiration criterion

        if (best_cost > cost_adm_s):  # update best sol if necessary
            best_s = np.copy(adm_s)  # update best solution
            best_cost = cost_adm_s

        # update tabu list by adding adm_s as a visited solution
        if not (tabu_list.full()):
            tabu_list.put(np.copy(adm_s))
        else:
            tabu_list.get()  # get an item to free a place
            tabu_list.put(np.copy(adm_s))

    total_it = total_it + 1

print('best sol, cost {} {}'.format(best_s, best_cost))