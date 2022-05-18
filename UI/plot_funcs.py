import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def create_plots(model, opt_pol, opt_rew, imdp, init, ax=None):
    """
    Creates some plots depending on model
    """
    if model=="1room heating":
        heatmap(opt_rew, (19,20), [36,40], [22.9,19.1], ax)
    if model=="UAV_gauss" or model=="UAV_dryden":
        plot_policy(opt_pol, init, imdp, ax)
    plt.show()

def heatmap(rewards, grid,xlim,ylim, ax=None):
    """
    Plots reachability heat map
    """
    if ax is None:
        fig=plt.figure()
        ax=plt.axes()
    ax.imshow(rewards[1:].reshape(grid), extent=xlim+ylim) 

def plot_policy(policy, init, imdp, ax=None):
    if ax is None:
        fig=plt.figure()
        ax = plt.axes(projection='3d')
    index = imdp.find_state_index(init.T)
    states = [imdp.States[index][:3]]
    for timestep in policy:
        index = timestep[index]
        states.append(imdp.States[index][:3])
    ax.plot3D(states)

def drone_plot(data, T, ax=None):
    """
    plots an updating 3D plot of drone position
    """
    plt.ion()
    x = [state[0][0] for state in data]
    y = [state[1][0] for state in data]
    z = [state[2][0] for state in data]
    if ax is None:
        fig=plt.figure()
        ax = plt.axes(projection='3d')
    #ax.set_xlim3d(min(x), max(x))
    #ax.set_ylim3d(min(y), max(y))
    #ax.set_zlim3d(0, max(z))
    for i in range(len(x)):
        ax.scatter3D(x[i], y[i], z[i], marker="*", c="blue")
        plt.pause(T)
    plt.show()

def noise(data):
    """
    plots 3D noise data (useful for visualising Dryden gust model)
    """
    plt.figure()
    for i in range(3):
        plt.plot([gust[i] for gust in data])
