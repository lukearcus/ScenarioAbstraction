import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import Dynamics
import drone_plotter
import StateSpace
import iMDP
import controller
import pickle

init_pos = np.ones((3,1))*1
init_vel = np.zeros((3,1))
init = np.concatenate((init_pos,init_vel))
T=1
test = Dynamics.Drone_dryden(init, T, 10)
control = np.ones((3,1))*1



ss = StateSpace.ContStateSpace(6, (tuple(0 for i in range(3)) + tuple(-5 for i in range(3)), [10 for i in range(3)]+[5 for i in range(3)]),
                                   [((2,2,2, -5, -5, -5),(3,4,3, 5, 5, 5))],
                                   [((9,9,9, -5, -5, -5),(10,10,10, 5, 5, 5))])

#with open('test_imdp_2.pkl', 'rb') as inp:
#    test_imdp = pickle.load(inp)
test_imdp = iMDP.iMDP(ss, test, (3,3,3,3,3,3), 100)

with open('test_imdp.pkl', 'wb') as outp:
    pickle.dump(test_imdp, outp, pickle.HIGHEST_PROTOCOL)
import pdb; pdb.set_trace()
cntrl = controller.controller(test_imdp, test)
states = [test.state]
gusts = []
for i in range(1000):
    if i%2 == 0:
        control = cntrl.get_action(test.state)
    test.state_update(control[3*(i%2):3*((i%2)+1)])
    states.append(test.state)
    gusts.append(np.copy(test.gusts))
ax = ss.draw_space([0,1,2])
drone_plotter.plot(states, T, ax)
