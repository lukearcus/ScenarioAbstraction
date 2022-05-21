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
hybrid_imdp = main.iMDP.hybrid_iMDP(hybrid_ss,hybrid_dyn,(20,20))

samples = 800
hybrid_imdp.update_probs(samples)


output_folder = 'output/'+type(hybrid_dyn).__name__+'/'+str(samples)
if not os.path.exists(output_folder):
    os.makedirs(output_folder+'/')
input_folder = 'input/'+type(hybrid_dyn).__name__+'/'+str(samples)
if not os.path.exists(input_folder):
    os.makedirs(input_folder+'/')
writer=main.iMDP.hybrid_PRISM_writer(hybrid_imdp, hybrid_dyn.horizon, input_folder, output_folder)
writer.write()
writer.solve_PRISM(12)
opt_pol, opt_delta, rew = writer.read()
import pdb; pdb.set_trace()
