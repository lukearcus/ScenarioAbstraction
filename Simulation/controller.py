import numpy as np

class controller:

    def __init__(self, iMDP, dyn):
        self.states = iMDP.States
        self.find_state_index = iMDP.find_state_index
        self.state_size = iMDP.part_size
        self.actions = {s: [] for s in self.states}
        for act in iMDP.Actions:
            for s in self.states:
                if s in iMDP.Actions[act]:
                    self.actions[s].append(act)
        self.actions = [self.actions[act][-1] if len(self.actions[act]) > 0 else (0,0,0,0,0,0) for act in self.actions]
        self.B_pinv = np.linalg.pinv(dyn.B_full_rank)
        self.A = dyn.A_full_rank

    def get_action(self, x):
        state_ind = self.find_state_index(x)
        if state_ind == len(self.states):
            return np.zeros((self.B_pinv.shape[0],1))
        sel_act = self.actions[self.states[state_ind]]
        goal_pos = sel_act
        acc = self.B_pinv @ (np.expand_dims(goal_pos,1) - self.A @ x)
        return acc

