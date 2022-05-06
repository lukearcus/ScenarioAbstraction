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
test = Dynamics.Full_Drone_dryden(init, T, 4, -4, 5)
control = np.ones((6,1))*1



ss = StateSpace.ContStateSpace(6, ((-15, -9, 5) + tuple(-2.25 for i in range(3)), [15, 9, 19]+[2.25 for i in range(3)]),
                                   [
                                       ((-11,-1,5, -2.25, -2.25, -2.25),(-5,3,15, 2.25, 2.25, 2.25)), # hole 1
                                       ((-11,5,5, -2.25, -2.25, -2.25),(-5,9,15, 2.25, 2.25, 2.25)),
                                       ((-11, 5, 13, -2.25, -2.25, -2.25),(-5, 9, 17, 2.25, 2.25, 2.25)),
                                       ((-11, 3, 5, -2.25, -2.25, -2.25),(-5, 5, 7, 2.25, 2.25, 2.25)),
                                       ((-1, 1, 5, -2.25, -2.25, -2.25),(3, 9, 11, 2.25, 2.25, 2.25)), # hole 2
                                       ((-1, 1, 15, -2.25, -2.25, -2.25),(3, 9, 17, 2.25, 2.25, 2.25)),
                                       ((-1, 1, 11, -2.25, -2.25, -2.25),(3, 3, 15, 2.25, 2.25, 2.25)),
                                       ((-1, 7, 11, -2.25, -2.25, -2.25),(3, 9, 15, 2.25, 2.25, 2.25)),
                                       ((-1, -3, 5, -2.25, -2.25, -2.25),(3, 1, 19, 2.25, 2.25, 2.25)), # tower
                                       ((3, -3, 5, -2.25, -2.25, -2.25),(9, 1, 11, 2.25, 2.25, 2.25)), # wall between
                                       ((-11, -5, 5, -2.25, -2.25, -2.25),(-7, -1, 13, 2.25, 2.25, 2.25)), # long obs
                                       ((-1, -9, 5, -2.25, -2.25, -2.25),(3, -3, 7, 2.25, 2.25, 2.25)),
                                       ((-1, -9, 15, -2.25, -2.25, -2.25),(3, -3, 19, 2.25, 2.25, 2.25)), # overhang
                                       ((11, -9, 5, -2.25, -2.25, -2.25),(15, -5, 7, 2.25, 2.25, 2.25)), # small last
                                       ((9, 5, 5, -2.25, -2.25, -2.25),(15, 9, 13, 2.25, 2.25, 2.25)), # next to goal

                                   ],
                                   [((11,1,5, -2.25, -2.25, -2.25),(15,5,9, 2.25, 2.25, 2.25))])

ax = ss.draw_space([0,1,2])
#with open('test_imdp.pkl', 'rb') as inp:
#    test_imdp = pickle.load(inp)
test_imdp = iMDP.iMDP(ss, test, (15,9,7,3,3,3), 100)
#test_imdp.create_probs(100)
#with open('test_imdp.pkl', 'wb') as outp:
#    pickle.dump(test_imdp, outp, pickle.HIGHEST_PROTOCOL)
import pdb; pdb.set_trace()
writer = iMDP.PRISM_writer(test_imdp, 64)
iMDP.solve_PRISM(writer.filename, writer.specification)
#cntrl = controller.controller(test_imdp, test)
#test = Dynamics.Drone_dryden(init, T, 10, -10, 5)
#states = [test.state]
#gusts = []
#for i in range(1000):
#    if i%2 == 0:
#        control = cntrl.get_action(test.state)
#    test.state_update(control[3*(i%2):3*((i%2)+1)])
#    states.append(test.state)
#    gusts.append(np.copy(test.gusts))
#ax = ss.draw_space([0,1,2])
#drone_plotter.plot(states, T, ax)
