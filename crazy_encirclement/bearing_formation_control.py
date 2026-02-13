import numpy as np
from icecream import ic
def bearing_based_formation_control(agents_pos, ego_pos, evader_pos, k,r):
    n_agents = agents_pos.shape[1]
    u = np.zeros(3)
    for i in range(n_agents):
        wij = (np.linalg.norm(agents_pos[:,i] - ego_pos) - r) /np.linalg.norm(agents_pos[:,i] - ego_pos)
        u[:,i] -= wij*(ego_pos - agents_pos[:,i])
        wit = (np.linalg.norm(ego_pos - evader_pos) - r/2) /np.linalg.norm(ego_pos[:,i] - evader_pos)
        u[:,i] -= wit*(ego_pos[:,i] - evader_pos)
    u = k*u
    return u