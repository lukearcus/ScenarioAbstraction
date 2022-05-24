import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pickle
import os

import main.Dynamics
import UI.plot_funcs
import main.StateSpace
import main.iMDP

init_state = np.array([[10,10]]).T
hybrid_dyn = main.Dynamics.multi_room_heating(init_state)
hybrid_ss = main.StateSpace.ContStateSpace(2, ((15, 15), (25, 25)), [], [((20, 20), (22, 22)) ])
hybrid_imdp = main.iMDP.hybrid_iMDP(hybrid_ss,hybrid_dyn,(40,40))

samples = 25
hybrid_imdp.update_probs(samples)


output_folder = 'output/'+type(hybrid_dyn).__name__+'/'+str(samples)
if not os.path.exists(output_folder):
    os.makedirs(output_folder+'/')
input_folder = 'input/'+type(hybrid_dyn).__name__+'/'+str(samples)
if not os.path.exists(input_folder):
    os.makedirs(input_folder+'/')
writer=main.iMDP.hybrid_PRISM_writer(hybrid_imdp, hybrid_dyn.horizon, input_folder, output_folder, True)
writer.write()
writer.solve_PRISM(12)
opt_pol, opt_delta, rew = writer.read()

fig, axs = plt.subplots(2)
axs[0].imshow(rew[2:3202:2].reshape(40,40), extent=[15,25,25,15])
axs[1].imshow(rew[3:3202:2].reshape(40,40), extent=[15,25,25,15])
plt.show()
