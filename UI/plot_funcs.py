import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def create_plots(model, opt_pol, opt_rew):
    """
    Creates some plots depending on model
    """
    if model=="1room heating":
        heatmap(opt_rew, (19,20), [36,40], [22.9,19.1])

def heatmap(rewards, grid,xlim,ylim):
    """
    Plots reachability heat map
    """
    ax = plt.imshow(rewards[1:].reshape(grid), extent=xlim+ylim)
    plt.show()

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
