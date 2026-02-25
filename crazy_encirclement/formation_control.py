import numpy as np
from icecream import ic
def formation_control(agents_dist, evader_dist, k,r):
    n_agents = agents_dist.shape[1]
    u = np.zeros(3)
    for i in range(n_agents):
        wij = (np.linalg.norm(agents_dist[:,i]) - r) /np.linalg.norm(agents_dist[:,i])
        u += wij*(agents_dist[:,i])
        wit = (np.linalg.norm(evader_dist) - r/1.5)/np.linalg.norm(evader_dist)
        u += wit*(evader_dist)
    u = k*u
    return u