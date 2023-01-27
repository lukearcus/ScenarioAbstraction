import numpy as np

class controller:

    def __init__(self, iMDPs, dyn, pol):
        self.states = iMDPs[0].States
        self.find_state_index = iMDPs[0].find_state_index
        self.state_size = iMDPs[0].part_size
        self.actions = {s: [] for s in self.states}
        
        self.policy = pol
        self.systems = dyn.individual_systems
        self.timestep = 0

    def get_action(self, x, r=0):
        state_ind = int(self.find_state_index(x.T)[0][0][0])
        sys = self.systems[r]
        B_pinv = np.linalg.pinv(sys.B_full_rank)
        A = sys.A_full_rank
        if state_ind == len(self.states):
            return np.zeros((B_pinv.shape[0],1))
        sel_act = self.policy[self.timestep][state_ind+1+r*(len(self.states)+1)]
        cont_act = int(sel_act[0])
        disc_act = int(sel_act[1])
        goal_pos = self.states[cont_act]
        acc = B_pinv @ (np.expand_dims(goal_pos,1) - A @ x)
        self.timestep+=1
        return acc, disc_act
