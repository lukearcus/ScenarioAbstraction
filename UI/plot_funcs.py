import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def create_plots(model, opt_pol, opt_rew, imdp, init, ax=None):
    """
    Creates some plots depending on model
    """
    if model=="1room heating":
        ax, im = heatmap(opt_rew, (19,20), [19.1,22.9], [36,40], ax)
        plt.colorbar(im);
        plt.xlabel("Zone Temp (" + '\u00b0' + "C)", fontsize=15)
        plt.ylabel("Radiator Temp (" + '\u00b0' + "C)", fontsize=15)
        plt.title("Our Result", fontsize=20)
    if  model=="conv_test":
        heatmap(opt_rew, (100,100), [-20,20], [20,-20], ax)
    if model.split("_")[0]=="UAV":
        plot_policy(opt_pol, init, imdp, ax)
    if model=="n_room_heating" or model=="steered_n_room_heating":
        nr_rooms = len(imdp.iMDPs)
        nr_states = len(imdp.iMDPs[0].States)+1
        fig, axs = plt.subplots(1, nr_rooms)
        for i in range(nr_rooms):
            axs[i], im = heatmap(opt_rew[nr_states*i:nr_states*(i+1)], (40,40), [20,25], [20,25], axs[i])
            axs[i].set_xlabel("Room 1 Temp (" + '\u00b0' + "C)", fontsize=15)
            axs[i].set_ylabel("Room 2 Temp (" + '\u00b0' + "C)", fontsize=15)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        if model=="n_room_heating":
            fig.suptitle("Separate Modes", fontsize=25)
        else:
            fig.suptitle("Steered Transitions", fontsize=25)
    if model=="steered_test" or model=="unsteered_test":
        nr_rooms = len(imdp.iMDPs)
        nr_states = len(imdp.iMDPs[0].States)+1
        fig, axs = plt.subplots(1, nr_rooms)
        for i in range(nr_rooms):
            axs[i], im = heatmap(opt_rew[nr_states*i:nr_states*(i+1)], (100,100), [-20,20], [-20,20], axs[i])
        if model=="unsteered_test":
            fig.suptitle("Unsteered Transitions", fontsize=25)
        else:
            fig.suptitle("Steered Transitions", fontsize=25)
    plt.show()

def heatmap(rewards, grid,xlim,ylim, ax=None):
    """
    Plots reachability heat map
    """
    if ax is None:
        fig=plt.figure()
        ax=plt.axes()
    im = ax.imshow(np.flipud(rewards[1:].reshape(grid).T), extent=xlim+ylim, vmin=0,vmax=1, cmap="jet")
    return ax, im

def plot_policy(policy, init, imdp_abstr, ax=None):
    if ax is None:
        fig=plt.figure()
        ax = plt.axes(projection='3d')
    counter = 0
    for imdp in imdp_abstr.iMDPs:
        ind = int(imdp.find_state_index(init[0].T)[0][0][0])
        x = [imdp.States[ind][0]]
        y = [imdp.States[ind][1]]
        z = [imdp.States[ind][2]]
        for timestep in policy:
            ind = int(timestep[ind+counter][0])
            if ind == -1:
                break
            x += [imdp.States[ind][0]]
            y += [imdp.States[ind][1]]
            z += [imdp.States[ind][2]]
        counter += len(imdp.States)+1
        ax.plot3D(x,y,z)

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
