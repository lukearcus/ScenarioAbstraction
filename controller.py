import numpy as np

class controller:

    def __init__(self, iMDP):
        self.states = iMDP.States
        self.actions = [act[-1]  for act in iMDP.Actions]
