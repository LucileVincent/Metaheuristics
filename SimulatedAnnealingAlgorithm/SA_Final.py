# load data from tsp file
import numpy as np
import math

tsp_data = np.loadtxt('gr17.2085.tsp')
print(tsp_data.shape)


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


def accept_solution(delta, T):
    decision = ['yes','no']
    prob = math.exp(-abs(delta)/T)  # probability of acceptance
    final = np.random.choice(decision, 1, p=[prob, 1-prob])  # select yes or no according to p
    if final == 'yes':
        return True
    else:
        return False


def geometric_cooling(T, alpha):
    return T*alpha


costs_hist = []
# random initial solution and its cost
s = np.random.choice(range(17), 17, replace=False)  # cities in [0,16]
cost_s = objectivefunction(s)

# save history of costs
costs_hist.append(cost_s)


# consider solution s as the best solution for now
best_s = np.copy(s)  # hard copy
best_cost = cost_s

# test with T= 10 and internal_it = 1
T = 500  # max temperature

while T>1:
    internal_it = 0
    while (internal_it <5):
        # pick a random neighbor and calculate its cost
        s_neighbor = pickneighbor(np.copy(s))
        cost_neighbor = objectivefunction(s_neighbor)

        # calculate difference of cost
        delta = cost_s - cost_neighbor

        if delta >=0: # neighbor is better than current solution s
            s = np.copy(s_neighbor) # accept new solution
            cost_s = cost_neighbor # save its cost
            # becareful here, need to check best_cost before update, since best_s can come from degraded solution
            if (best_cost > cost_s):
                best_s = np.copy(s_neighbor) # update best solution
                best_cost = cost_neighbor
                costs_hist.append(cost_neighbor) # save history of costs
        else: # current solution s is still better than its neighbor
            if (accept_solution(delta, T)): # if Ture then accept solution degradation
                s = np.copy(s_neighbor)
                cost_s = cost_neighbor
                costs_hist.append(cost_neighbor)

        internal_it = internal_it +1

    T = geometric_cooling(T, 0.99)

print ('best sol, cost {} {}'.format(best_s,best_cost))

import matplotlib.pyplot as plt
plt.plot(costs_hist, 'b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# convert distance matrix between cities into coordinates of cities
from sklearn import manifold  # multidimensional scaling
mds_model = manifold.MDS(n_components=2, random_state=17, dissimilarity='precomputed')
mds_fit = mds_model.fit(tsp_data)
mds_coords = mds_model.fit_transform(tsp_data)

# plot a random solution over cities
s = np.random.choice(range(17), 17, replace=False)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(mds_coords[s, 0], mds_coords[s, 1], 'b-o')
plt.plot([mds_coords[s[0], 0], mds_coords[s[-1], 0]],
         [mds_coords[s[0], 1], mds_coords[s[-1], 1]], 'b-o')

# plot the best solution over cities
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(mds_coords[best_s, 0], mds_coords[best_s, 1], 'b-o')
plt.plot([mds_coords[best_s[0], 0], mds_coords[best_s[-1], 0]],
         [mds_coords[best_s[0], 1], mds_coords[best_s[-1], 1]], 'b-o')

