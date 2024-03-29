import numpy as np
import main.BAS_params as BAS_class

class dynamic_base:
    """
    Base class defining dynamics
    """
    robust = False
    hybrid=False
    horizon = 64
    grouped_timesteps = 1
    Q = 0
    def __init__(self, init_state):
        self.state = init_state

    def state_update(self):
        """
        update state
        """
        pass

    def noise(self,state=None):
        """
        Add noise
        """
        pass


class hybrid_dynamic_base:
    """
    Base class for hybdrid dynamic systems
    """
    horizon = 64
    grouped_timesteps = 1
    N_modes=-1
    robust=False
    hybrid = True
    steered = False
    
    def __init__(self, init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, true_trans_mat, N_modes):
        self.state = init_state
        self.mode = init_mode
        self.T=T
        self.N_modes = N_modes
        
        self.transition_matrix = transition_matrix
        self.true_trans_mat = true_trans_mat
        #self.true_trans_mat = np.zeros((transition_matrix.shape[0],transition_matrix.shape[1]))
        #for m, mode_trans in enumerate(transition_matrix):
        #    for m_next_id, trans_prob in enumerate(mode_trans):
        #        if trans_prob.size != 1:
        #            if m_next_id+1 == self.true_trans_mat.shape[1]:
        #                self.true_trans_mat[m,m_next_id] = \
        #                    1-np.sum(self.true_trans_mat[m,:]) # we assume thi
        #            else:
        #                self.true_trans_mat[m,m_next_id] = \
        #                    np.random.uniform(trans_prob[0],trans_prob[1])
        self.individual_systems = [LTI_gauss(init_state, A[i], B[i], Q[i], u_max[i], u_min[i], sigma[i]) for i in range(2)]


    def state_update(self, control):
        
        # update continuous state
        curr_dyn = self.individual_systems[self.mode]
        curr_dyn.state_update(control)
        self.state = curr_dyn.state

        # update discrete state
        self.mode = np.random.choice(self.N_modes, p = self.true_trans_mat[self.mode])
        
        # store current state in individual system state
        self.individual_systems[self.mode].state = self.state

    def noise(self):
        """
        Add noise
        """
        pass

class single_hybrid(hybrid_dynamic_base):

    def __init__(self, system):
        self.individual_systems = [system]
        self.state = system.state
        self.N_modes = 1
        self.T=system.T
        self.mode = 0
        self.transition_matrix = np.array([[1]])
        self.true_trans_mat = np.array([[1]])
        if hasattr(system, 'mu'):
            self.mu = [system.mu]
            self.sigma = [system.sigma]
        else:
            self.noises = [self.noise]

class steered_base(hybrid_dynamic_base):

    steered = True
    def state_update(self, cont_control, disc_control):
        # update continuous state
        curr_dyn = self.individual_systems[self.mode]
        curr_dyn.state_update(cont_control)
        self.state = curr_dyn.state

        # update discrete state
        self.mode = np.random.choice(self.N_modes, p = self.true_trans_mat[self.mode][disc_control])
        
        # store current state in individual system state
        self.individual_systems[self.mode].state = self.state

class steered_MC(hybrid_dynamic_base):
    """
    Wrapper class for converting a non-steered MC hybrid model to a dummy steered MDP
    (i.e. MDP with one action per state)
    """
    steered = True
    def __init__(self, MC_model):
       self.state = MC_model.state
       self.mode = MC_model.mode
       self.T = MC_model.T
       self.N_modes = MC_model.N_modes
       self.individual_systems = MC_model.individual_systems
       self.transition_matrices = [np.expand_dims(probs,0) for probs in MC_model.transition_matrix]
       self.true_trans_mats = [np.expand_dims(probs,0) for probs in MC_model.true_trans_mat]
        

    def state_update(self, cont_control, disc_control=0):
        # update continuous state
       
        curr_dyn = self.individual_systems[self.mode]
        curr_dyn.state_update(cont_control)
        self.state = curr_dyn.state

        # update discrete state
        
        self.mode = np.random.choice(self.N_modes, p = self.true_trans_mats[self.mode].flatten())
        
        # store current state in individual system state
        self.individual_systems[self.mode].state = self.state

class multi_room_heating(hybrid_dynamic_base):
    """
    Multiple room heating, 1 heater shared between all rooms
    same A matrix in all modes
    """
    horizon=32
    def __init__(self, init_state, init_mode=0, T=15, min_u=0, max_u=1, nr_rooms = 2, sigma=0.25):
        u_min = [np.ones((nr_rooms,1))*min_u for i in range(nr_rooms)]
        u_max = [np.ones((nr_rooms,1))*max_u for i in range(nr_rooms)]
        ambient_temp = 6
        sigma = [np.diag([sigma for i in range(nr_rooms)]) for i in range(nr_rooms)] # noise is equal in all modes
        if nr_rooms == 2:
            a_12 = 0.022
            b_1 = 0.0167
            b_2 = 0.0167
            c_1=0.8
            c_2=0.8
            A = [np.array([[1-b_1-a_12, a_12],[a_12, 1-b_2-a_12]]) for i in range(nr_rooms)]
            B = [np.array([[c_1,0],[0,c_2/2]]).T, np.array([[c_1/2,0],[0, c_2]]).T]
            Q = [np.array([[b_1*ambient_temp, b_2*ambient_temp]]).T for i in range(nr_rooms)]
            transition_matrix = np.array([[(0.4,0.6), (0.4,0.6)],[(0.4,0.6), (0.4,0.6)]])
            true_trans_mat = np.array([[0.5,0.5],[0.5,0.5]])
        elif nr_rooms == 3:
            a_12 = 0.022
            a_13 = 0.022
            a_23 = 0.001
            A = [np.array([[1, a_12, a_13],[a_12, 1, a_23],[a_13, a_23, 1]]) for i in range(nr_rooms)]
        super().__init__(init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, true_trans_mat, nr_rooms)


class unsteered_test(hybrid_dynamic_base):
    def __init__(self, init_state, init_mode, sigma=0.1):
        u_min = [np.ones((2,1))*-2 for i in range(2)]
        u_max = [np.ones((2,1))*2 for i in range(2)]
        sigma = [np.diag([sigma for i in range(2)]) for i in range(2)] # assume noise is equal in all modes
        ambient_temp = 6
        a = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])]
        b = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
        q = [np.array([[0,0]]).T for i in range(2)]
        transition_matrix = np.array([[0.5, 0.5],[0.5,0.5]])
        
        super.__init__(init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, nr_rooms)

class steered_test(steered_base):
    """
    test with arbitrary dynamics
    """
    def __init__(self, init_state, init_mode, sigma=0.1):
        self.transition_matrices = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]
        self.state=init_state
        self.mode=init_mode
        self.T=0.1
        self.N_modes = 2
        u_min = [np.ones((2,1))*-2 for i in range(2)]
        u_max = [np.ones((2,1))*2 for i in range(2)]
        sigma = [np.diag([sigma for i in range(2)]) for i in range(2)] # assume noise is equal in all modes
        ambient_temp = 6
        a = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])]
        b = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
        q = [np.array([[0,0]]).T for i in range(2)]
        #self.transition_matrix = np.array([[0.5, 0.5],[0.5,0.5]])
        self.individual_systems = [LTI_gauss(init_state, a[i], b[i], q[i], u_max[i], u_min[i], sigma[i]) for i in range(2)]        

class steered_drone_speed(steered_base):
    """
    2 Drone systems with different allowed accelerations and noises
    """
    def __init__(self, init_state, init_mode=0, T=1):
        self.state = init_state
        self.mode = init_mode
        self.individual_systems = [Full_Drone_gauss(init_state, T, 4, -4, _sigma=0.15),\
                                   Full_Drone_gauss(init_state, T, 8, -8, _sigma=0.3)]
        self.transition_matrices = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]

class steered_drone_dir(steered_base):
    """
    2 Drone systems with different allowed accelerations and noises
    """
    def __init__(self, init_state, init_mode=0, T=1):
        self.state = init_state
        self.mode = init_mode
        self.T = T
        self.individual_systems = [Full_Drone_gauss(init_state, T, np.array([[8,8,4,8,8,4]]).T,\
                                    np.array([[-8,-8,-4,-8,-8,-4]]).T, _sigma=0.15),\
                                   Full_Drone_gauss(init_state, T, np.array([[4,4,8,4,4,8]]).T,\
                                    np.array([[-4,-4,-8,-4,-4,-8]]).T, _sigma=0.15)]
        self.transition_matrices = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]

class drone_var_noise(hybrid_dynamic_base):
    def __init__(self, init_state, init_mode=0, T=1):
        self.state = init_state
        self.mode = init_mode
        self.T = T
        self.individual_systems = [Full_Drone_gauss(init_state, T, 4,\
                                    -4, _sigma=0.15),\
                                   Full_Drone_gauss(init_state, T, 4,\
                                    -4, _sigma=1.5)]
        self.transition_matrix = np.array([[0.9,0.1],[0.3,0.7]])


class steered_multi_room(multi_room_heating):
    """
    Multiple room heating but now with control over discrete modes
    """
    steered = True
    def __init__(self, init_state, init_mode=0, T=15, min_u=0, max_u=1, nr_rooms = 2, sigma=0.25):
        super().__init__(init_state, init_mode, T, min_u, max_u, nr_rooms, sigma)
        if nr_rooms == 2:
            # now one transition matrix for each mode, each row is an action and contains transition probabilities
            # to next mode
            self.transition_matrices = [np.array([[(0.8,0.99),(0.01,0.2)],[(0.01,0.2),(0.8,0.99)]]) for i in range(2)]
            self.true_trans_mats = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]
    def state_update(self, cont_control, disc_control):
        # update continuous state
        curr_dyn = self.individual_systems[self.mode]
        curr_dyn.state_update(cont_control)
        self.state = curr_dyn.state

        # update discrete state
        self.mode = np.random.choice(self.N_modes, p = self.true_trans_mats[self.mode][disc_control])
        
        # store current state in individual system state
        self.individual_systems[self.mode].state = self.state

class robust_test(hybrid_dynamic_base):
    robust = True

    def __init__(self, init_state, sigma):
        self.T = 1
        self.state = init_state
        self.mode = 0
        self.A_mats = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])] 
        self.B_mats = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
        self.Q_mats = [np.array([[0,0]]).T for i in range(2)]
        #A_mats = [np.array([[0.5, 0],[0,1]]),np.array([[1,0],[0,0.5]])]
        #B_mats = [np.eye(2),np.eye(2)]
        #Q_mats = [np.zeros((2,1)), np.zeros((2,1))]
        self.u_max = [np.ones((2,1))*5 for i in range(2)]
        self.u_min = [np.ones((2,1))*-5 for i in range(2)]
        self.transition_matrix = [np.random.random((2,2)) for i in range(2)]
        self.true_trans_mat = np.copy(self.transition_matrix)
        self.sigma = [np.diag([sigma for i in range(2)]) for i in range(2)]
        self.mu = [np.zeros((2,1)) for i in range(2)]
        self.individual_systems = [LTI_gauss(init_state, self.A_mats[i], self.B_mats[i], self.Q_mats[i], self.u_max[i], self.u_min[i], self.sigma[i]) for i in range(2)]        

class multi_room_heating_robust(hybrid_dynamic_base):
    """
    Multiple room heating, 1 heater shared between all rooms
    same A matrix in all modes
    """
    robust=True
    horizon=32
    def __init__(self, init_state, init_mode=0, T=15, min_u=0, max_u=1, nr_rooms = 2, sigma=0.25):
        self.state = init_state
        self.mode = init_mode
        self.u_min = [np.ones((nr_rooms,1))*min_u for i in range(nr_rooms)]
        self.u_max = [np.ones((nr_rooms,1))*max_u for i in range(nr_rooms)]
        ambient_temp = 6
        self.sigma = [np.diag([sigma for i in range(nr_rooms)]) for i in range(nr_rooms)] # noise is equal in all modes
        self.mu = [np.zeros((nr_rooms,1)) for i in range(nr_rooms)]
        if nr_rooms == 2:
            a_12 = 0.022
            b_1 = 0.0167
            b_2 = 0.0167
            c_1=0.8
            c_2=0.8
            self.A_mats = [np.array([[1-b_1-a_12, a_12],[a_12, 1-b_2-a_12]]) for i in range(nr_rooms)]
            self.B_mats = [np.array([[c_1,0],[0,c_2/2]]).T, np.array([[c_1/2,0],[0, c_2]]).T]
            self.Q_mats = [np.array([[b_1*ambient_temp, b_2*ambient_temp]]).T for i in range(nr_rooms)]
            self.transition_matrix = np.array([[0.5, 0.5],[0.5,0.5]])
            self.true_trans_mat = np.copy(self.transition_matrix)
        else:
            raise NotImplementedError
        super().__init__(init_state, init_mode, T, self.A_mats, self.B_mats, self.Q_mats, self.u_max, self.u_min, self.sigma, self.transition_matrix, nr_rooms)

class separate_robust_test(hybrid_dynamic_base):
    """
    test with arbitrary dynamics
    """
    def __init__(self, init_state, sigma=0.1):
        self.state=init_state
        self.mode=0
        self.T=0.1
        self.N_modes = 2
        u_min = [np.ones((2,1))*-2 for i in range(2)]
        u_max = [np.ones((2,1))*2 for i in range(2)]
        sigma = np.diag([sigma for i in range(2)]) # assume noise is equal in all modes
        ambient_temp = 6
        a = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])]
        b = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
        q = [np.array([[0,0]]).T for i in range(2)]
        self.transition_matrix = np.array([[0.9999, 0.0001],[0.0001,0.9999]])
        self.true_trans_mat = np.copy(self.transition_matrix)
        self.individual_systems = [LTI_gauss(init_state, a[i], b[i], q[i], u_max[i], u_min[i], sigma) for i in range(2)]        

class LTI_gauss(dynamic_base):
    """
    Basic LTI class which takes matrices as inputs
    """
    def __init__(self, init, _A, _B, _Q, _u_max, _u_min, _sigma):
        self.A = _A
        self.B = _B
        self.Q = _Q
        self.u_max = _u_max
        self.u_min = _u_min
        self.sigma = _sigma
        self.state = init
        self.mu = np.zeros(self.state.shape)
        if _A.shape[1] <= _B.shape[1]:
            self.A_full_rank = _A
            self.B_full_rank = _B
            self.grouped_timesteps=1
        else:
            self.grouped_timestpes = int(math.ceil(_A.shape[1]/_B.shape[1]))
            As = []
            Bs = []
            for i in range(self.grouped_timesteps):
                As += np.linalg.matrix_power(_A, i+1)
                Bs += np.linalg.matrix_power(_A, i)*_B

            As.reverse()
            Bs.reverse()

            self.A_full_rank = np.hstack(tuple(As))
            self.B_full_rank = np.hstack(tuple(Bs))

    def state_update(self, control):
        self.state = self.A@self.state + self.B@control + self.Q + self.noise()

    def noise(self):
        return np.random.multivariate_normal(self.mu.flatten(), self.sigma)[:,np.newaxis]

class heat_1_room(dynamic_base):
    """
    1 room BAS heating
    """
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
        """
        Checks if control input is valid
        """
        split = self.split_control(control)
        for control in split:
            if np.any(control > self.u_max) or np.any(control < self.u_min):
                return False
        return True

    def split_control(self, control):
        """
        Splits up vector of multiple control inputs
        """
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
        return np.random.multivariate_normal(self.mu.flatten(), self.sigma)[:,np.newaxis]

class Drone_base(dynamic_base):
    """
    UAV base class
    """
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
    """
    Base class for fully actuate drone system (achieved by grouping 2 timesteps together)
    """
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
        if type(max_acc) == int:
            self.u_max = np.ones((self.B.shape[1],1)) * max_acc
            self.u_min = np.ones((self.B.shape[1],1)) * min_acc
        else:
            self.u_max = max_acc
            self.u_min = min_acc

class Drone_gauss(Drone_base):
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), _mu=0, _sigma=1):
        self.mu = np.ones((6,1))*_mu
        self.sigma = np.diag(np.ones((6))*_sigma)
        super().__init__(init_state, T,max_acc, min_acc)

    def noise(self):
        return np.random.multivariate_normal(self.mu.flatten(), self.sigma)[:,np.newaxis]

class Full_Drone_gauss(Full_Drone_Base):
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), _mu=0, _sigma=1):
        self.mu = np.ones((6,1))*_mu
        self.sigma = np.diag(np.ones((6))*_sigma)
        super().__init__(init_state, T,max_acc, min_acc)

    def noise(self):
        return self.A_1_step @ np.random.multivariate_normal(self.mu.flatten(), self.sigma)[:,np.newaxis]+ np.random.multivariate_normal(self.mu.flatten(), self.sigma)[:,np.newaxis]



class Drone_dryden(Drone_base):
    """
    Assumes low-altitude dryden model
    """
    def __init__(self, init_state, T,max_acc = float('inf'), min_acc = -float('inf'), W20=15):
        self.gusts = np.random.random((3,1))
        self.sigma_w = 0.1*W20
        super().__init__(init_state, T, max_acc, min_acc)

    def noise(self):
        """
        Dryden noise model
        """
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
        return self.A_1_step @ self.single_noise(self.state) + self.single_noise(self.state)

    def single_noise(self, state):
        sigma_w = self.sigma_w
        h = float(state[2])/0.3048
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

        V = np.linalg.norm(state[3:])
        T = self.T

        pos_noise = np.zeros((3,1))
        self.gusts[0] = (1-V*T/L_u)*np.copy(self.gusts[0]) + np.sqrt(2*V*T/L_u)*sigma_u*np.random.standard_normal()
        self.gusts[1] = (1-V*T/L_v)*np.copy(self.gusts[1]) + np.sqrt(2*V*T/L_v)*sigma_v*np.random.standard_normal()
        self.gusts[2] = (1-V*T/L_w)*np.copy(self.gusts[2]) + np.sqrt(2*V*T/L_w)*sigma_w*np.random.standard_normal()
        return np.concatenate((pos_noise, self.gusts))

