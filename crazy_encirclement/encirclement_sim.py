from embedding_SO3_sim import Embedding
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 13
})
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.spatial.transform import Rotation as R
from icecream import ic
import pandas as pd
import os
import pickle 

N = 200
r = 1
k_phi = 5
kx = 5
kv = 6.5*np.sqrt(2)
n_agents = 3
phi_dot = 0.6#np.deg2rad(35)
dt = 0.1
save = False

mb = 0.04
g = 9.81
I3 = np.array([0,0,1]).T.reshape(3)
w_r = 0 #reference yaw
ca_1 = np.array([np.cos(w_r),np.sin(w_r),0]).T #auxiliar vector 
Ca_r = np.zeros((3,3,n_agents,N))
Ca_r[:,:,0,0] = np.eye(3)
quat = np.zeros((4,n_agents,N))

ra_r = np.zeros((3,n_agents,N))
va_r = np.zeros((3,n_agents,N))
va_r_dot = np.zeros((3,n_agents,N))
accels = np.zeros((3,n_agents,N))
phi_cur = np.zeros((n_agents, N))    
phi_dot_cur = np.zeros((n_agents, N))

agents_r = np.zeros((3,n_agents, N))
agents_v = np.zeros((3,n_agents, N))
if n_agents >1:
    n_diff = int(math.factorial(n_agents) / (math.factorial(2) * math.factorial(n_agents-2)))
else:
    n_diff = 1
phi_diff =  np.zeros((n_diff,N))
distances = np.zeros((n_diff,N))
va_r_dot = np.zeros((3,n_agents,N))
Ca_r = np.zeros((3,3,n_agents,N))
identity_matrix = np.eye(3)
Ca_b = np.tile(
    identity_matrix[:, :, np.newaxis, np.newaxis], (1, 1, n_agents, N))
va_r = np.zeros((3,n_agents,N))
f_T_r = np.zeros((n_agents,N))
angles = np.zeros((3,n_agents,N))
Wr_r = np.zeros((3,n_agents,N))

agents_r[:, 0, 0] = 1*np.array([r*np.cos(np.deg2rad(10)),r*np.sin(np.deg2rad(10)),0.6]).T
agents_r[:, 1, 0] = 1*np.array([r*np.cos(np.deg2rad(100)),r*np.sin(np.deg2rad(100)),0.6]).T
agents_r[:, 2, 0] = 1.*np.array([r*np.cos(np.deg2rad(200)),r*np.sin(np.deg2rad(200)) ,0.6]).T
# agents_r[:, 3, 0] = 1.*np.array([r*np.cos(np.deg2rad(290)),r*np.sin(np.deg2rad(290)) ,0.6]).T

ra_r[:,:,0] = agents_r[:,:,0]
for i in range(n_agents):
    phi_cur[i,0] = np.mod(np.arctan2(agents_r[1,i,0],agents_r[0,i,0]),2*np.pi)

embedding = Embedding(r, phi_dot,k_phi, 'dumbbell',n_agents,agents_r[:,:,0],dt)

for i in range(0,N-1):
    print("percentage: ", float(i/N))

    phi_new, target_r_new, target_v_new, phi_diff_new, distances_new,debug = embedding.targets(agents_r[:,:,i],i)

    #ic(target_r_new)
    phi_cur[:,i+1] = phi_new
    # phi_dot_cur[:,i] = (phi_cur[:,i+1] - phi_cur[:,i])/dt
    ra_r[:,:,i+1] = target_r_new#*np.random.uniform(0.99,1.01)
    # va_r[:,:,i+1] = target_v_new#*np.random.uniform(0.99,1.01)

    va_r[:,:,i+1] = ((ra_r[:,:,i+1] - ra_r[:,:,i])/(dt))#*np.random.uniform(0.8,1.2)
    # va_r_dot[:,:,i] = (va_r[:,:,i+1] - va_r[:,:,i])/dt
    phi_diff[:,i] = phi_diff_new
    distances[:,i] = distances_new
    #ic(va_r[:,:,i+1])

    accels[:,:,i] =  kx*(ra_r[:,:,i+1] - agents_r[:,:,i]) + kv*(va_r[:,:,i+1] - agents_v[:,:,i]) # +
    agents_v[:,:,i+1] =( agents_v[:,:,i] + accels[:,:,i]*dt)*np.random.uniform(0.99,1.01)
    agents_r[:,:,i+1] = (agents_r[:,:,i] + agents_v[:,:,i]*dt + 0.5*accels[:,:,i]*dt**2)*np.random.uniform(0.99,1.01)
    #agents_r[:,:,i+1] = target_r_new
#saving data to a pickle file
with open ('positions_new_controller.pkl','wb') as f:
    pickle.dump(agents_r,f)

#saving data to a pickle file
with open ('positions.pkl','wb') as f:
    pickle.dump(agents_r,f)

figures_dir = "figures/"
os.makedirs(figures_dir, exist_ok=True)
t = np.linspace(0, N*dt, N)
colors = plt.cm.viridis(np.linspace(0, 1, n_agents))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
legends = []
for agent in range(1):
    color = colors[agent]
    ax.plot3D(ra_r[0,agent,0:-1], ra_r[1,agent,0:-1], ra_r[2,agent,0:-1],color=color,label=f"Des Pose agent {agent+1}")
    # ax.scatter(agents_r[0,agent,0], agents_r[1,agent,0], agents_r[2,agent,0],color=color,marker='o')
    # ax.scatter(agents_r[0,agent,-1], agents_r[1,agent,-1], agents_r[2,agent,-1],color='black',marker='o')
    ax.plot3D(agents_r[0,agent,1:-1], agents_r[1,agent,1:-1], agents_r[2,agent,1:-1],color=color, linestyle='dashed', label=f"Real Pose {agent+1}")

ax.legend()    #legends.append(f"Real trajectory agent {agent+1}")
#, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
ax.set_xlabel('X Axis (m)')
ax.set_ylabel('Y Axis (m)')
ax.set_zlabel('Z Axis (m)')
# ax.set_zlim(0.59,0.61)
#ax.view_init(elev=90, azim=-90)

if save:
# # List of perspectives to save (elev, azim)
    angles = [
        (20, 30),   # Low angle, front-left
        (60, 45),   # High angle, front-left
        (45, 90),   # Side view
        (90, 0),    # Directly above
        (10, 180),  # Low angle, back view
        (75, 210),  # High angle, back-right
        (30, 270),  # Side view, right
        (60, 315)   # High angle, front-right
    ]
    plt.savefig(f"{figures_dir}/3_agents_SO3_default_view.pdf", bbox_inches='tight', pad_inches=0.1)
    # Save the plot from each perspective
    for j, (elev, azim) in enumerate(angles):
        if j == 3:
            ax.legend(loc='lower left')
            ax.set_zlabel('')
        else:
            ax.legend()
            ax.set_zlabel('Z Axis (m)')
        ax.view_init(elev=elev, azim=azim)  # Set the view
        plt.savefig(f"{figures_dir}/3_agents_SO3_view_{j+1}.pdf", bbox_inches='tight', pad_inches=0.1)  # Save with unique filename

    # Show the plot in the last perspective if needed
    #plt.show()
    plt.close()
else:
    plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for agent in range(n_agents):
#     ax.plot3D(agents_r[0,agent,1:-1], agents_r[1,agent,1:-1], agents_r[2,agent,1:-1])

# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# plt.savefig()
# for i in range(n_agents):
#     plt.subplot(3,1,1)
#     #plt.title(f"Velocities agent {i+1}")
#     plt.plot(t[0:-1],va_r[0,i,0:-1])
#     plt.ylabel("$v_x$ (m/s)")
#     plt.subplot(3,1,2)
#     plt.plot(t[0:-1],va_r[1,i,0:-1])
#     plt.ylabel("$v_y$ (m/s)")
#     plt.subplot(3,1,3)
#     plt.plot(t[0:-1],va_r[2,i,0:-1])
#     plt.ylabel("$v_z$ (m/s)")
#     plt.xlabel("Time (s)")
    
#     #plt.show()
#     if save:
#         plt.savefig(f"{figures_dir}/velocities_agent_{i+1}.pdf")
#         plt.close()
#     else:
#         plt.show()

for i in range(n_agents):
    plt.subplot(3,1,1)
    #plt.title(f"Positions agent {i+1}")
    plt.plot(t[0:-1],ra_r[0,i,0:-1],color=colors[i])
    plt.plot(t[0:-1],agents_r[0,i,0:-1],linestyle='dashed',color=colors[i])
    plt.legend([f"Des Pos {i+1}",f"Actual Pos {i+1}"])
    plt.ylabel("X axis (m)")
    plt.subplot(3,1,2)
    plt.plot(t[0:-1],ra_r[1,i,0:-1],color=colors[i])
    plt.plot(t[0:-1],agents_r[1,i,0:-1],linestyle='dashed',color=colors[i])
    
    plt.ylabel("Y axis (m)")
    plt.subplot(3,1,3)
    plt.plot(t[0:-1],ra_r[2,i,0:-1],color=colors[i])
    plt.plot(t[0:-1],agents_r[2,i,0:-1],linestyle='dashed',color=colors[i])

    plt.ylabel("Z axis (m)")
    plt.xlabel("Time (s)")

    if save:
        plt.savefig(f"{figures_dir}/positions_agent_{i+1}.pdf")
        plt.close()
    else:
        plt.show()
for i in range(n_agents):
    plt.plot(t[0:-1],np.rad2deg(phi_cur[i,0:-1]),label=f"Angle agent {i+1}")
plt.ylabel("Angles (degrees)")
plt.xlabel("Time (s)")
#plt.title("Angles of the agents")
plt.legend()
if save:
    plt.savefig(f"{figures_dir}/angles.pdf")
    plt.close()
else:
    plt.show()

plt.plot(t[0:-1],distances[0,0:-1],label=f"Distance agents 1-2")
plt.plot(t[0:-1],distances[1,0:-1],label=f"Distance agents 1-3")
plt.plot(t[0:-1],distances[2,0:-1],label=f"Distance agents 2-3")
plt.ylabel("Distances (m)")
plt.xlabel("Time (s)")
#plt.title("Distances between agents")
plt.legend()
if save:
    plt.savefig(f"{figures_dir}/distances.pdf")
    plt.close()
else:
    plt.show()
for i in range(n_diff):
    plt.plot(t[0:-1],distances[i,0:-1],label=f"Distance agent {i+1}")
plt.ylabel("Distances (m)")
plt.xlabel("Time (s)")
plt.title("Distances between agents")
plt.legend()
if save:
    plt.savefig(f"{figures_dir}/distances.png")
    plt.close()
else:
    plt.show()

for i in range(n_diff):
    plt.plot(t[0:-1],np.rad2deg(phi_diff[i,0:-1]),label=f"Phase difference agent {i+1}")
plt.title("Phase differences between agents")
plt.ylabel("$\phi$ (degrees)")
plt.xlabel("Time (s)")
plt.legend()
if save:
    plt.savefig(f"{figures_dir}/phase_diff.pdf")
    plt.close()
else:
    plt.show()

for agent in range(n_agents):
    plt.subplot(3,1,1)
    #plt.title(f"Positions x, y, and z errors of all the agents")
    plt.ylabel(f"$e_x$ (m)")
    plt.plot(t[0:-1],ra_r[0,agent,0:-1]-agents_r[0,agent,0:-1],label=f"agent {agent+1}")
    plt.subplot(3,1,2)
    plt.ylabel(f"$e_y$ (m)")
    plt.plot(t[0:-1],ra_r[1,agent,0:-1]-agents_r[1,agent,0:-1],label=f"agent {agent+1}")
    plt.subplot(3,1,3)
    plt.ylabel(f"$e_z$ (m)")
    plt.plot(t[0:-1],ra_r[2,agent,0:-1]-agents_r[2,agent,0:-1],label=f"agent {agent+1}")
    plt.xlabel("Time (s)")

# Add legend for each subplot after plotting all agents
plt.subplot(3, 1, 1)
plt.legend()
plt.subplot(3, 1, 2)
plt.legend()
plt.subplot(3, 1, 3)
plt.legend()

    
if save:
    plt.savefig(f"{figures_dir}/error_agent.pdf")
    plt.close()
else:
    plt.show()





