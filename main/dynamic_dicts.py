import numpy as np



def multi_room_heating(init_state, init_mode=0, T=15, min_u=0, max_u=1, nr_rooms = 2, sigma=0.25):
    """
    Multiple room heating, 1 heater shared between all rooms
    same A matrix in all modes
    """
    horizon=32         
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
        transition_matrix = np.array([[0.5, 0.5],[0.5,0.5]])
    elif nr_rooms == 3:
        a_12 = 0.022
        a_13 = 0.022
        a_23 = 0.001
        A = [np.array([[1, a_12, a_13],[a_12, 1, a_23],[a_13, a_23, 1]]) for i in range(nr_rooms)]
    return init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, nr_rooms

def unsteered_test(init_state, init_mode, sigma=0.1):
    u_min = [np.ones((2,1))*-2 for i in range(2)]
    u_max = [np.ones((2,1))*2 for i in range(2)]
    sigma = [np.diag([sigma for i in range(2)]) for i in range(2)] # assume noise is equal in all modes
    a = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])]
    b = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
    q = [np.array([[0,0]]).T for i in range(2)]
    transition_matrix = np.array([[0.5, 0.5],[0.5,0.5]])
    N_modes = 2
    return init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, N_modes

def steered_test(self, init_state, init_mode, sigma=0.1):
    """
    test with arbitrary dynamics
    """
    init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, N_modes = unsteered_test(init_state, init_mode, sigma)
    transition_matrices = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]
    return init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, N_modes
    
def steered_multi_room(self, init_state, init_mode=0, T=15, min_u=0, max_u=1, nr_rooms = 2, sigma=0.25):
    """
    Multiple room heating but now with control over discrete modes
    """
    steered = True
    init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, N_modes = multi_room_heating(init_state, init_mode, T, min_u, max_u, nr_rooms, sigma)
    transition_matrices = [np.array([[0.9,0.1],[0.1,0.9]]) for i in range(2)]
    return init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrices, nr_rooms

def robust_test(init_state):
    T = 1
    A_mats = [np.array([[1, 0],[0, 1]]), np.array([[1, 0.5],[0.5, 1]])] 
    B_mats = [np.array([[1,0],[0,1]]), np.array([[1,0],[0, 1]])]
    Q_mats = [np.array([[0,0]]).T for i in range(2)]
    #A_mats = [np.array([[0.5, 0],[0,1]]),np.array([[1,0],[0,0.5]])]
    #B_mats = [np.eye(2),np.eye(2)]
    #Q_mats = [np.zeros((2,1)), np.zeros((2,1))]
    u_max = [np.ones((2,1))*5 for i in range(2)]
    u_min = [np.ones((2,1))*-5 for i in range(2)]
    sigma = [np.diag([sigma for i in range(2)]) for i in range(2)]
    return init_state, None, T, A, B, Q, u_max, u_min, sigma, None, 2


def robust_separate_test(init_state, sigma=0.1):
    """
    test with arbitrary dynamics
    """
    init_state, _, T, A, B, Q, u_max, u_min, sigma, __, N_modes = robust_test(init_state)
    init_mode=0
    transition_matrix = np.array([[0.9999, 0.0001],[0.0001,0.9999]])
    return init_state, init_mode, T, A, B, Q, u_max, u_min, sigma, transition_matrix, N_modes

def heat_1_room(self, init, T=15, min_u = [14, -10], max_u = [28,10]):
    """
    1 room BAS heating
    """
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
        return np.random.multivariate_normal(self.mu, self.sigma)
