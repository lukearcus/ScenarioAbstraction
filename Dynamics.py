import numpy as np
import BAS_params as BAS_class

class dynamic_base:

    grouped_timesteps = 1
    def __init__(self, init_state):
        self.state = init_state

    def state_update(self):
        # update state
        pass

    def noise(self):
        pass

class heat_1_room(dynamic_base):
    def __init__(self, init, T=15, min_u = [14, -10], max_u = [28,10]):
        self.state=init
        self.T = T
        self.u_min = np.array([min_u]).T
        self.u_max = np.array([max_u]).T
        BAS = BAS_class.parameters()

        Tswb    = BAS.Boiler['Tswbss'] - 20
        Twss    = BAS.Zone1['Twss']
        Pout1   = BAS.Radiator['Zone1']['Prad']      
        
        w       = BAS.Radiator['w_r']
        
        BAS.Zone1['Cz'] = BAS.Zone1['Cz']
        
        m1      = BAS.Zone1['m'] # Proportional factor for the air conditioning
        
        k1_a    = BAS.Radiator['k1']
        k0_a    = BAS.Radiator['k0'] #Proportional factor for the boiler temp. on radiator temp.
        
        # Defining Deterministic Model corresponding matrices
        A_cont      = np.zeros((2,2));
        A_cont[0,0] = -(1/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))-((Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])) - ((m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz']))
        A_cont[0,1] = (Pout1*BAS.Radiator['alpha2'] )/(BAS.Zone1['Cz'])
        A_cont[1,0] = (k1_a)
        A_cont[1,1] = -(k0_a*w) - k1_a
        
        B_cont      = np.zeros((2,2))
        B_cont[0,0] = (m1*BAS.Materials['air']['Cpa'])/(BAS.Zone1['Cz'])
        B_cont[1,1] = (k0_a*w) # < Allows to change the boiler temperature

        
        W_cont  = np.array([
                [ (Twss/(BAS.Zone1['Rn']*BAS.Zone1['Cz']))+ (BAS.Radiator['alpha1'])/(BAS.Zone1['Cz']) ],
                [ (k0_a*w*Tswb) ],
                ])
        
        self.A = np.eye(2) + self.T*A_cont
        self.B = B_cont*self.T
        self.Q = W_cont*self.T
        
        self.B_full_rank = self.B
        # Determine system dimensions
        self.n = np.size(self.A,1)
        self.p = np.size(self.B,1)

        self.mu = np.array([0,0])
        self.sigma = np.diag([ BAS.Zone1['Tz']['sigma'], BAS.Radiator['rw']['sigma'] ])

    def is_valid(self, control):
        split = self.split_control(control)
        for control in split:
            if np.any(control > self.u_max) or np.any(control < self.u_min):
                return False
        return True
    
    def split_control(self, control):
        in_dims = self.B.shape[1]
        num_ins = int(control.size/in_dims)
        return np.split(control, num_ins)
    
    def state_update(self, control):
        split_controls = self.split_control(control)
        for control in split_controls:
            if not self.is_valid(control):
                raise NotImplementedError
            self.state = self.A @ self.state + self.B @ control + self.Q+ self.noise()

    def noise(self):
        return np.random.multivariate_normal(self.mu, self.sigma)

class Drone_base(dynamic_base):

    def __init__(self, init_state, T, max_acc = float('inf'), min_acc = -float('inf')):
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
        self.Q = np.zeros((6,1))

        self.u_max = np.ones((self.B.shape[1],1)) * max_acc
        self.u_min = np.ones((self.B.shape[1],1)) * min_acc
        self.A_full_rank = self.A @ self.A
        self.B_full_rank = np.concatenate((self.A @ self.B, self.B),1)
    

    def is_valid(self, control):
        split = self.split_control(control)
        for control in split:
            if np.any(control > self.u_max) or np.any(control < self.u_min):
                return False
        return True

    def split_control(self, control):
        in_dims = self.B.shape[1]
        num_ins = int(control.size/in_dims)
        return np.split(control, num_ins)

    def state_update(self, control):
        split_controls = self.split_control(control)
        for control in split_controls:
            if not self.is_valid(control):
                raise NotImplementedError
                #control = (control*self.u_max)/np.linalg.norm(control) # bounds the input

            if not self.crashed:
                self.state = self.A @ self.state + self.B @ control + self.noise()

    def noise(self):
        return 0

class Full_Drone_Base(Drone_base):
    grouped_timesteps = 2
    def __init__(self, init_state, T, max_acc = float('inf'), min_acc = -float('inf')):
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

        self.A_full_rank = self.A @ self.A
        self.B_full_rank = np.concatenate((self.A @ self.B, self.B), 1)
        self.A_1_step = np.copy(self.A)
        self.A = self.A_full_rank
        self.B = self.B_full_rank
        self.u_max = np.ones((self.B.shape[1],1)) * max_acc
        self.u_min = np.ones((self.B.shape[1],1)) * min_acc


class Drone_gauss(Drone_base):
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), _mu=0, _sigma=1):
        self.mu = _mu
        self.sigma = _sigma
        super().__init__(init_state, T,max_acc, min_acc)

    def noise(self):
        return np.random.normal(self.mu, self.sigma, (6,1))

class Full_Drone_gauss(Full_Drone_Base):
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), _mu=0, _sigma=1):
        self.mu = _mu
        self.sigma = _sigma
        super().__init__(init_state, T,max_acc, min_acc)

    def noise(self):
        return self.A_1_step @ np.random.normal(self.mu, self.sigma, (6,1)) + np.random.normal(self.mu, self.sigma, (6,1))


class Drone_dryden(Drone_base):
    """
    Assumes low-altitude dryden model
    """
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), W20=15):
        self.gusts = np.random.random((3,1))
        self.sigma_w = 0.1*W20
        super().__init__(init_state, T, max_acc, min_acc)
    
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
       
        pos_noise = np.zeros((3,1))
        
        self.gusts[0] = (1-V*T/L_u)*np.copy(self.gusts[0]) + np.sqrt(2*V*T/L_u)*sigma_u*np.random.standard_normal()
        self.gusts[1] = (1-V*T/L_v)*np.copy(self.gusts[1]) + np.sqrt(2*V*T/L_v)*sigma_v*np.random.standard_normal()
        self.gusts[2] = (1-V*T/L_w)*np.copy(self.gusts[2]) + np.sqrt(2*V*T/L_w)*sigma_w*np.random.standard_normal()
        return np.concatenate((pos_noise, self.gusts))


class Full_Drone_dryden(Full_Drone_Base):
    """
    Assumes low-altitude dryden model
    """
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), W20=15):
        self.gusts = np.random.random((3,1))
        self.sigma_w = 0.1*W20
        super().__init__(init_state, T, max_acc, min_acc)
    
    def noise(self):
        return self.A_1_step @ self.single_noise() + self.single_noise()

    def single_noise(self):
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
       
        pos_noise = np.zeros((3,1))
        self.gusts[0] = (1-V*T/L_u)*np.copy(self.gusts[0]) + np.sqrt(2*V*T/L_u)*sigma_u*np.random.standard_normal()
        self.gusts[1] = (1-V*T/L_v)*np.copy(self.gusts[1]) + np.sqrt(2*V*T/L_v)*sigma_v*np.random.standard_normal()
        self.gusts[2] = (1-V*T/L_w)*np.copy(self.gusts[2]) + np.sqrt(2*V*T/L_w)*sigma_w*np.random.standard_normal()
        return np.concatenate((pos_noise, self.gusts))

