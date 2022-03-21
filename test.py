import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import Dynamics
import drone_plotter
import StateSpace
import iMDP
import controller

init_pos = np.ones((3,1))*100
init_vel = np.zeros((3,1))
init = np.concatenate((init_pos,init_vel))
T=1
test = Dynamics.Drone_dryden(init, T, 10)
control = np.ones((3,1))*1



ss = StateSpace.ContStateSpace(6, (tuple(0 for i in range(3)) + tuple(-5 for i in range(3)), [10 for i in range(3)]+[5 for i in range(3)]),
                                   [((200,200,200, 500, 500, 500),(300,400,300, 1000, 1000, 1000))],
                                   [((900,900,900, 0, 0, 0),(1000,1000,1000, 500, 500, 500))])

import pdb; pdb.set_trace()
test_imdp = iMDP.iMDP(ss, test, (3,3,3,3,3,3), 100)

import pdb; pdb.set_trace()
cont = controller.controller(test_imdp)
states = [test.state]
gusts = []
#for i in range(1000):
#    test.state_update(control)
#    states.append(test.state)
#    if i == 30:
#        control = np.zeros((3,1))
#    gusts.append(np.copy(test.gusts))
#ax = ss.draw_space([0,1,2])
#drone_plotter.plot(states, T, ax)
