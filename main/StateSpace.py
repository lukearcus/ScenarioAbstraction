import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

class ContStateSpace:
    """
    Object defining a multidimensional state space, including critical and goal regions
    """
    def __init__(self, n_dims_, valid_range_, unsafes_, goals_):
        '''
        inputs:
            number of dimensions in state sapce
            range of valid states as pair of Nd tuples (i.e. one with minimums, other with maximums)
            unsafes - list of tuples with each one defining a rectangular unsafe region in state space
            goals - same as unsafes but with goal regions
        '''

        # Do some checks on inputs here!!

        self.n_dims = n_dims_
        self.valid_range = np.array(valid_range_)

        self.unsafes = np.array(unsafes_)
        if len(self.unsafes.shape) == 2:
            self.unsafes = np.expand_dims(self.unsafes, 0)
        self.goals = np.array(goals_)
        if len(self.goals.shape) == 2:
            self.goals = np.expand_dims(self.goals, 0)


    def in_set(self, x, check_set):
        """
        Checks if a given point, x, is within a given set
        """
        x = np.array(x)
        in_range = [x <= check_set[1], x >= check_set[0]]
        return np.all(in_range)

    def check_valid(self, x):
        """
        Checks if x is in the valid range
        """
        return self.in_set(x, self.valid_range)

    def check_safe(self, x):
        """
        Checks if x is valid & within the safe areas of the space
        """
        if self.check_valid(x):
            for unsafe in self.unsafes:
                if self.in_set(x, unsafe):
                    return False
            return True
        else:
            return False

    def check_goal(self, x):
        """
        Checks if x is in a goal region
        """
        if self.check_valid(x):
            for goal in self.goals:
                if self.in_set(x, goal):
                    return True
            return False
        else:
            return False

    def min_max_to_3d_points(self, rect, chosen_dims):
        """
        Converts a minimum and maximum point to a 3d cuboid
        """
        X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
            [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]

        pos = tuple(rect[0][i] for i in chosen_dims)
        size = tuple(rect[1][i]-rect[0][i] for i in chosen_dims)
        X = np.array(X).astype(float)
        for i in range(3):
            X[:,:,i] *= size[i]
        X += np.array(pos)
        return X

    def draw_space(self, chosen_dims = 0):
        """
        Draws the chosen dimensions of the state space
        """
        if chosen_dims == 0:
            chosen_dims = [i for i in range(self.n_dims)]
        if len(chosen_dims) > 3:
            print("Cannot draw more than 3D, please specify chosen dimensions")
            return False
        else:
            if len(chosen_dims) == 2:
                plt.figure()
                ax = plt.gca()
                for unsafe in self.unsafes:
                    size = unsafe[1] - unsafe[0]
                    ax.add_patch(Rectangle(unsafe[0], size[0], size[1], color='r', alpha=0.5))
                for goal in self.goals:
                    size = goal[1] - goal[0]
                    ax.add_patch(Rectangle(goal[0], size[0], size[1], color='g', alpha=0.5))

                plt.xlim(self.valid_range[0][chosen_dims[0]], self.valid_range[1][chosen_dims[0]])
                plt.ylim(self.valid_range[0][chosen_dims[1]], self.valid_range[1][chosen_dims[1]])
                return ax

            if len(chosen_dims) == 3:
                fig=plt.figure()
                ax = plt.axes(projection='3d')
                ax.set_xlim3d(self.valid_range[0][chosen_dims[0]], self.valid_range[1][chosen_dims[0]])
                ax.set_ylim3d(self.valid_range[0][chosen_dims[1]], self.valid_range[1][chosen_dims[1]])
                ax.set_zlim3d(self.valid_range[0][chosen_dims[2]], self.valid_range[1][chosen_dims[2]])
                g = []
                colours = []
                for unsafe in self.unsafes:
                    g.append(self.min_max_to_3d_points(unsafe, chosen_dims))
                    colours.append("red")
                for goal in self.goals:
                    g.append(self.min_max_to_3d_points(goal, chosen_dims))
                    colours.append("green")
                cubes = Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colours,6), alpha=0.5)
                ax.add_collection3d(cubes)
                return ax


