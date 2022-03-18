import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import Dynamics
import drone_plotter
import StateSpace

init_pos = np.ones((3,1))*100
init_vel = np.zeros((3,1))
init = np.concatenate((init_pos,init_vel))
T=0.1
test = Dynamics.Drone_dryden(init, T)
control = np.ones((3,1))*1

states = [test.state]
gusts = []
for i in range(1000):
    test.state_update(control)
    states.append(test.state)
    if i == 30:
        control = np.zeros((3,1))
    gusts.append(np.copy(test.gusts))


ss = StateSpace.ContStateSpace(6, (tuple(0 for i in range(6)),tuple(1000 for i in range(6))),
                                   [((200,200,200, 500, 500, 500),(300,400,300, 1000, 1000, 1000))],
                                   [((900,900,900, 0, 0, 0),(1000,1000,1000, 500, 500, 500))])
ax = ss.draw_space([0,1,2])
drone_plotter.plot(states, T, ax)
