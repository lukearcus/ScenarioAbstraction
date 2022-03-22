import numpy as np

class controller:

    def __init__(self, iMDP, dyn):
        self.states = iMDP.States
        self.find_state_index = self.iMDP.find_state_index
        self.state_size = iMDP.part_size
        self.actions = [iMDP.Actions[act][-1]  for act in iMDP.Actions]
        self.B_pinv = np.linalg.pinv(dyn.B_full_rank)
        self.A = dyn.A_full_rank

    def get_action(self, x):
        state_ind = self.find_state_index(x)
        if state_ind > len(self.states):
            return np.zeros((self.B_pinv.shape[0],1))
        sel_act = self.actions[state_ind]
        goal_pos = sel_act + self.state_size/2
        acc = self.B_pinv @ (np.expand_dims(goal_pos,1) - self.A @ x)
        return acc

