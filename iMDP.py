import numpy as np
import itertools

class iMDP:

    def __init__(self, Cont_state_space, Dynamics, parts, num_samples):
        """
        Const_state_space is StateSpace object
        parts is a list/ tuple of partitions for each dimension
        """

        self.N_d = Cont_state_space.n_dims
        self.ss = Cont_state_space
        self.dyn = Dynamics
        min_pos = Cont_state_space.valid_range[0]
        self.part_size = (Cont_state_space.valid_range[1]-Cont_state_space.valid_range[0])/parts
        increments = []
        for dim in range(self.N_d):
            dim_inc = []
            curr_inc = float(min_pos[dim])
            for i in range(parts[dim]):
                dim_inc.append(curr_inc)
                curr_inc += self.part_size[dim]
            increments.append(dim_inc)
        self.States = list(itertools.product(*increments))
        self.Goals = self.find_goals()
        self.Unsafes = self.find_unsafes()
       
        self.max_acc_vecs = np.zeros((self.dyn.B.shape))
        for i in range(self.dyn.B.shape[1]):
            control = np.zeros((self.dyn.B.shape[1], 1))
            control[i] = self.dyn.u_max
            self.max_acc_vecs[:,i] = (self.dyn.B @ control).flatten()

        self.A_pinv = np.linalg.pinv(self.dyn.A)

        self.B_pinv = np.linalg.pinv(self.dyn.B_full_rank) 
        np_incs = np.stack(([0 for i in range(self.N_d)],self.part_size)).T
        incs = [list(inc) for inc in np_incs]
        self.edge_diffs = list(itertools.product(*incs))
        self.Actions = self.determine_actions()

    def determine_actions(self):
        actions = {i : [] for i in self.States}
        for i, state in enumerate(self.States):
            if i % 100 == 0:
                print(i/len(self.States))
            centre = state + self.part_size/2
            backwards = self.backward_states(centre)
            for prev_state in backwards:
                actions[prev_state].append(state)
        return actions




    def find_unsafes(self):
        """
        Finds all iMDP states that overlap with unsafe regions
        Will expand unsafe regions somewhat
        (also assumes everything is rectangular)
        """
        unsafes = [state for state in self.States if self.check_unsafe(state)]
        return unsafes

    def check_unsafe(self, state):
        overlap = True
        for i in range(self.N_d):
            for unsafe in self.ss.unsafes:
                overlap = overlap and state[i] < unsafe[1][i]
                overlap = overlap and state[i]+self.part_size[i] > unsafe[0][i]
        return overlap

    def find_goals(self):
        """
        finds all iMDP states which are fully enclosed in the goal region
        This effectively shrinks the goal region but should be fine
        (also assumes everything is rectangular)
        """
        goals = [state for state in self.States if self.ss.check_goal(state)
                and self.ss.check_goal(state+self.part_size)]
        return goals

    def backward_states(self, d_j):
        """
        Assuming backward reachable set is convex (I think it is??)

        Needs to be sped up - once we have found an unreachable state we know that all states beyond that state are also unreachable!
        """
        max_change = np.zeros((self.N_d))
        for i, max_vec in enumerate(self.max_acc_vecs.T):
            max_change_curr = np.abs(self.A_pinv @ (d_j - max_vec) - d_j)

            max_change = np.maximum(max_change_curr, max_change)


        reachable_states = []
        for state in self.States:
            reachable = True
            for diff in self.edge_diffs:
                edge = np.array(state) + np.array(diff)
                if np.any(np.abs(edge-d_j) > max_change):
                    reachable = False
                    #if not self.is_reachable(d_j, edge):
                    #reachable = False
                    break
            if reachable: reachable_states.append(state)
        return reachable_states

    def is_reachable(self, d_j, x):
        u = self.B_pinv @ (d_j- self.dyn.A_full_rank @ x)
        return self.dyn.is_valid(u)
