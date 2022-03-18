import numpy as np

class dynamic_base:

    def __init__(self, init_state):
        self.state = init_state

    def state_update(self):
        # update state
        pass

    def noise(self):
        pass

class Drone_base(dynamic_base):

    def __init__(self, init_state, T):
        self.state = init_state
        self.T = T
        self.crashed = False
        self.A = np.array([[1, 0, 0, T, 0, 0],
                           [0, 1, 0, 0, T, 0],
                           [0, 0, 1, 0, 0, T],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        self.B = np.array([[(T**2)/2, 0, 0],
                           [0, (T**2)/2, 0],
                           [0, 0, (T**2)/2],
                           [T, 0, 0],
                           [0, T, 0],
                           [0, 0, T]])

    def state_update(self, control):
        if not self.crashed:
            self.state = self.A @ self.state + self.B @ control + self.noise()

    def noise(self):
        return 0

class Drone_gauss(Drone_base):
    def __init__(self, init_state, T, _mu=0, _sigma=1):
        self.mu = _mu
        self.sigma = _sigma
        super().__init__(init_state, T)

    def noise(self):
        return np.random.normal(self.mu, self.sigma, (6,1))

class Drone_dryden(Drone_base):
    """
    Assumes low-altitude dryden model
    """
    def __init__(self, init_state, T, W20=15):
        self.gusts = np.random.random((3,1))
        self.sigma_w = 0.1*W20
        super().__init__(init_state,T)
    
    def noise(self):
        sigma_w = self.sigma_w
        h = float(self.state[2])/0.3048
        if h < 0:
            self.crashed = True
            return np.zeros((6,1))
        sigma_u = sigma_w/(0.177+0.000823*h)**0.4
        sigma_v = sigma_u

        if h <= 0:
            h = 0.01
        L_w = h
        L_u = h/(0.177+0.000823*h)**1.2
        L_v = L_u

        V = np.linalg.norm(self.state[3:])
        T = self.T
       
        self.gusts[0] = (1-V*T/L_u)*np.copy(self.gusts[0]) + np.sqrt(2*V*T/L_u)*sigma_u*np.random.standard_normal()
        self.gusts[1] = (1-V*T/L_v)*np.copy(self.gusts[1]) + np.sqrt(2*V*T/L_v)*sigma_v*np.random.standard_normal()
        self.gusts[2] = (1-V*T/L_w)*np.copy(self.gusts[2]) + np.sqrt(2*V*T/L_w)*sigma_w*np.random.standard_normal()
        pos_noise = np.zeros((3,1)) 
        return np.concatenate((pos_noise, self.gusts))

