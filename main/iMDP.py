import numpy as np
import itertools
from scipy.stats import beta
from scipy.spatial import Delaunay
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import time
import math


def dec_round(num, decimals):
    power = 10**decimals
    return math.trunc(power*num)/power


class iMDP:
    """
    iMDP abstraction class
    """
    MAX_MEM=16 # max memory allowance in GB
    def __init__(self, Cont_state_space, Dynamics, parts, _beta = 0.01):
        """
        Const_state_space is StateSpace object
        Dynamics is a Dynamics object defining the model dynamics, including control limits
        parts is a list/ tuple of partitions for each dimension
        _beta is the confidence level
        """
        self.beta = _beta
        self.N_d = Cont_state_space.n_dims
        self.ss = Cont_state_space
        self.dyn = Dynamics
        min_pos = Cont_state_space.valid_range[0]
        self.part_size = (Cont_state_space.valid_range[1]-Cont_state_space.valid_range[0])/parts
        state_start = time.perf_counter()
        self.States, self.corner_array, self.end_corners_flat = self.create_partition(min_pos, parts)
        self.all_corners_flat = self.corner_array.reshape((-1,self.N_d))
        state_end = time.perf_counter()
        print("States set up in "+str(state_end-state_start))
        self.adj_states = self.build_adj_states(parts) # this is slow
        adj_end  = time.perf_counter()
        print("Adjacency set up in "+str(adj_end-state_end))
        self.Goals = self.find_goals()
        self.Unsafes = self.find_unsafes()
        self.A_pinv = np.linalg.pinv(self.dyn.A)
        self.B_pinv = np.linalg.pinv(self.dyn.B_full_rank)

        np_incs = np.stack(([0 for i in range(self.N_d)],self.part_size)).T
        act_start = time.perf_counter()
        self.Actions, self.Actions_forward, self.enabled_actions = self.determine_actions()
        act_end = time.perf_counter()
        print("Actions set up in "+str(act_end-act_start))

    def update_probs(self, N):
        """
        Updates probabilities with new number of samples, N
        """
        self.trans_probs, self.trans_ids = self.determine_probs(N) # can we not throw away samples?

    def build_adj_states(self,parts):
        """
        Builds list of lists, each list contains states adjacent to the state at the corresponding index
        """
        adj_states = [[] for state in self.States]
        for i, _ in enumerate(self.States):
            step = 1
            checker = 1
            for dim in range(self.N_d):
                checker *= parts[-1-dim]
                if i%checker != 0:
                    adj = i-step
                    if adj >= 0:
                        adj_states[i].append(adj)
                if (i+step)%checker != 0:
                    adj = i+step
                    if adj < len(self.States):
                        adj_states[i].append(adj)
                step *= parts[-1-dim]
        #incs = np.vstack((np.diag(self.part_size),np.diag(-self.part_size)))
        #adj_array = np.array(self.States)[:, np.newaxis, :] + incs
        #adj_states = [[self.States.index(tuple(adj_state)) for adj_state in adj\
        #                if self.check_in_valid_range(adj_state)[0]]\
        #                for adj in tqdm(adj_array)]
        return adj_states

    def create_partition(self, min_pos, parts):
        """
        Creates partitions
        returns:
            states:- a list of States (each entry being the centre point of a state)
            corner_array:- numpy array of corners for all states
            end_corners_flat:- flattened array of only end corners for each state (i.e. top right & bottom left)
        """
        state_increments = []
        print("Setting up iMDP states")
        dim_counter = [0 for _ in range(self.N_d)]
        for dim in range(self.N_d):
            dim_inc = []
            state_inc = []
            curr_inc = float(min_pos[dim])
            for i in range(parts[dim]):
                state_inc.append(curr_inc+self.part_size[dim]/2)
                dim_inc.append(curr_inc)
                curr_inc += self.part_size[dim]
            dim_inc.append(curr_inc)
            state_increments.append(state_inc)
        states = list(itertools.product(*state_increments))
        part_incs = [[-part_size/2, part_size/2] for part_size in self.part_size]
        part_arr = np.array(list(itertools.product(*part_incs)))
        corner_array = np.array(states)[:, np.newaxis, :] + part_arr
        end_corners_flat = np.reshape(corner_array[:,[0,-1],:],(-1,self.N_d))
        coord_list = [list(range(parts[i])) for i in range(self.N_d)]
        self.N_d_coords = list(itertools.product(*coord_list))
        return states, corner_array, end_corners_flat

    def create_table(self, N):
        """
        Creates lookup table based on N samples
        row corresponds to the number of samples not in a partition
        first entry is lower bound second is upper bound
        """
        print("Building lookup table")
        beta_bar = self.beta/(2*N)
        table = np.zeros((N+1, 2))
        table[N,0] = 0
        table[N,1] = 1-beta.ppf(beta_bar, N-1, 1)
        for k in tqdm(range(N)):
                id_in = (k-1) // 1 +1
                table[k,0] = 1 - beta.ppf(1-beta_bar, id_in+1, N-id_in) # N-(d-k)+1 == N-k since d == 1
                if k == 0:
                    table[k,1] = 1
                else:
                    id_out = id_in - 1
                    table[k,1] = 1 - beta.ppf(beta_bar, id_out+1, N-id_out)
        return table

    def determine_probs(self, N):
        """
        Enumerates over actions to find the probability of arriving in the goal state associated with that action
        """
        self.lookup_table = self.create_table(N)
        print("Finding transition probabilities")
        probs = {act:[] for act in self.enabled_actions}
        ids = dict(probs)
        for action in tqdm(self.enabled_actions):
            ids[action], probs[action] = self.comp_bounds(self.States[action], N)
        return probs, ids

    def comp_bounds(self, action, N):
        """
        compute probability bounds by adding noise to goal position and checking the resulting state
        """
        resulting_states = self.add_noise(action, N)
        inds = self.find_state_index_with_init(resulting_states, action)
        N_in = [inds.count(i) for i in range(len(self.States)+1)]
        probs = self.samples_to_prob(N, N_in)
        arr = np.array([[i,p] for i, p in enumerate(probs) if p != -1], dtype='object')
        return list(arr[:,0]), list(arr[:,1])

    def add_noise(self, action, N):
        """
        Adds noise to a chosen target state
        either by generating samples from a normal distribution or using noise function from dynamics
        """
        p = len(action)
        if hasattr(self.dyn, 'mu'):
            resulting_states = np.random.multivariate_normal(action, self.dyn.sigma, (N))
        else:
            resulting_states = np.zeros((N,p))
            self.dyn.state = np.expand_dims(pos,1)
            for i in range(N):
                resulting_states[i,:] = (np.expand_dims(pos,1) + self.dyn.noise()).T
        return resulting_states

    def samples_to_prob(self, N, N_in):
        """
        Takes samples as a list and returns probabilites for each state
        """
        probs = [-1 for state in self.States]
        nonzeros = [i for i, e in enumerate(N_in[:-1]) if e != 0]
        for j in nonzeros:
            k = N-N_in[j]
            probs[j] = [max(1e-4,dec_round(self.lookup_table[k,0],5)),\
                        dec_round(self.lookup_table[k,1],5)]

        k_deadlock = N_in[-1]
        probs.append([max(1e-4,dec_round(1-self.lookup_table[k_deadlock,1],5)),\
                      dec_round(1-self.lookup_table[k_deadlock,0],5)])
        return probs

    def find_state_index_with_init(self, x, start):
        """
        Finds the state index from a starting state index by expanding outwards
        """
        inds = [i for i in range(x.shape[0])]
        valids = self.check_in_valid_range(x)
        test_states = x[valids,:]
        results = [-1 if valid else len(self.States) for valid in valids]
        inds = [inds[i] for i, valid in enumerate(valids) if valid]
        start_ind = self.States.index(start)
        test_set = [start_ind]
        tested = []
        while -1 in results:
            curr_found = self.find_state_index(test_states, test_set)
            for i, found in enumerate(curr_found):
                if found != -1:
                    results[inds[i]] = found
            inds = [inds[i] for i, found in enumerate(curr_found) if found == -1]
            test_states = x[inds,:]
            tested += test_set
            next_test = []
            for test_ind in test_set:
                to_add = self.adj_states[test_ind]
                to_add = [elem for elem in to_add if elem not in tested]
                next_test += to_add
            test_set = list(set(next_test))
        return results

    def check_in_valid_range(self, x):
        """
        checks if a point is in the defined state space
        """
        if len(x.shape) < 2:
            test_pos = x[np.newaxis, np.newaxis, :]
            num_ins = 1
        else:
            test_pos = x[:, np.newaxis, :]
            num_ins = x.shape[0]
        shape =  (num_ins,1,2,self.corner_array.shape[2])
        end_corners = self.corner_array[[0,-1],[0,-1],:] # could use end_corners_flat and access relevant bits
        flat_ends = np.reshape(end_corners,(-1,self.N_d))
        signs = (np.sign(flat_ends-test_pos)+1).astype('bool')
        reshaped = signs.reshape(shape)
        summed = np.logical_xor(reshaped[:,:,0,:],reshaped[:,:,1,:])
        abs_sum = np.all(summed,2)
        return [np.any(abs_sum[i,:]) for i in range(num_ins) ]

    def find_state_index(self, x, state_inds_to_check = 'all'):
        """
        Finds state indices of sample points x (Num_samplesxN_d array)
        state_inds_to_check defines which states we look at, either all or a specific subset
        If providing a specific subset to look at state_inds_to_check should be a list of indices
        """
        if len(x.shape) < 2:
            test_pos = x[np.newaxis, np.newaxis, :]
            num_ins = 1
        else:
            test_pos = x[:, np.newaxis, :]
            num_ins = x.shape[0]
        if state_inds_to_check == 'all':
            shape =  (num_ins,self.corner_array.shape[0],2,self.corner_array.shape[2])
            signs = (np.sign(self.end_corners_flat-test_pos)+1).astype('bool')
            reshaped = signs.reshape(shape)
            summed = np.logical_xor(reshaped[:,:,0,:],reshaped[:,:,1,:])
            abs_sum = np.all(summed,2)
            list_out =  [np.where(abs_sum[i,:]) if np.any(abs_sum[i,:]) else len(self.States) for i in range(num_ins) ]
        else:
            shape = (num_ins,len(state_inds_to_check),2,self.corner_array.shape[2])
            first = lambda x: 2*x
            second = lambda x: 2*x+1
            indices = [f(ind) for ind in state_inds_to_check for f in (first, second)]
            corners_to_check = self.end_corners_flat[indices,:]
            signs = (np.sign(corners_to_check-test_pos)+1).astype('bool')
            reshaped = signs.reshape(shape)
            summed = np.logical_xor(reshaped[:,:,0,:],reshaped[:,:,1,:])
            abs_sum = np.all(summed,2)
            list_out =  [state_inds_to_check[np.where(abs_sum[i,:])[0][0]] if np.any(abs_sum[i,:]) else -1 for i in range(num_ins) ]
        return  list_out

    def determine_actions(self):
        """
        Determines iMDP actions, if dimensions are unequal might have some issues
        """
        print("Determining actions")
        u = [[self.dyn.u_min[i], self.dyn.u_max[i]] for i in range(len(self.dyn.u_max))]
        x_inv_area = np.zeros((2**len(self.dyn.u_max), self.dyn.A.shape[0]))
        for i, u_elem in enumerate(itertools.product(*u)):
            list_elem = list(u_elem)
            x_inv_area[i,:] = (self.A_pinv @ (self.dyn.B @ np.array(list_elem) + self.dyn.Q)).flatten()
        corners_array = self.all_corners_flat

        dim_equal = self.dyn.A.shape[0] == self.dyn.B.shape[1]
        if dim_equal:
            n = self.dyn.A.shape[0]
            u_avg = np.array(self.dyn.u_max + self.dyn.u_min)/2
            u_avg = u_avg.T
            u = np.tile(u_avg, (n, 1)) + np.diag((self.dyn.u_max.T - u_avg)[0])

            origin = self.A_pinv @ (self.dyn.B @ np.array(u_avg).T)

            basis_vectors = np.zeros((n,n))
            for i, elem in enumerate(u):
                point = self.A_pinv @ (self.dyn.B @ np.expand_dims(elem,1))
                basis_vectors[i,:] = point.flatten() - origin.flatten()
            parallelo2cube = np.linalg.inv(basis_vectors) # changed inv to pinv (is this allowed????)
            x_inv_area_normalised = x_inv_area @ parallelo2cube
            predSet_originShift = -np.average(x_inv_area_normalised, axis=0)
            allRegionVertices = corners_array @ parallelo2cube - predSet_originShift
            # use basis vectors
        else:
            x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
            allRegionVertices = corners_array
        actions_forward = {i : [] for i, s in enumerate(self.States) if i not in self.Unsafes}
        actions_inv = {i : [] for i,s in enumerate(self.States) if i not in self.Unsafes}
        if dim_equal:
            nr_acts = len(actions_inv)
            for act in tqdm(actions_inv):
                A_inv_d = self.A_pinv @ np.array(self.States[act])
                all_vert_normed = (A_inv_d @ parallelo2cube) - allRegionVertices

                poly_reshape = np.reshape(all_vert_normed, (len(self.States),n*(2**n)))
                enabled_in = np.maximum(np.max(poly_reshape, axis=1), -np.min(poly_reshape, axis=1)) <= 1.0
                state_ids = np.where(enabled_in)[0]
                actions_inv[act] = [state_id for state_id in state_ids\
                                if state_id not in self.Unsafes]
                for state_id in actions_inv[act]:
                    actions_forward[state_id].append(act)
        else:
            for act in tqdm(actions_inv):
                state_counter = {i: 0 for s_num, i in enumerate(self.States) if s_num not in self.Unsafes}
                A_inv_d = self.A_pinv @ np.array(act)
                all_vertices = A_inv_d - corners_array
                in_hull = x_inv_hull.find_simplex(all_vertices) >= 0
                if np.any(in_hull):
                    for val in corners_array[in_hull]:
                        for state in self.Corners_to_states[tuple(val)]:
                            if self.States.index(state) not in self.Unsafes:
                                state_counter[state]+=1
                                if state_counter[state] == 2**self.N_d:
                                    actions[act].append(state)
                state_ids = np.where(enabled_in)[0]
                actions_inv[act] = [state_id for state_id in state_ids if state_id not in self.Unsafes]
         
        enabled = [act for act in actions_inv if len(actions_inv[act]) > 0]
        return actions_inv, actions_forward, enabled

    def find_unsafes(self):
        """
        Finds all iMDP states that have a centre in unsafe region
        """
        unsafes = [i for i, state in enumerate(self.States) if not self.ss.check_safe(state)]
        return unsafes

    def find_goals(self):
        """
        checks if centre of state is a goal region
        """
        goals = [i for i, state in enumerate(self.States) if self.ss.check_goal(state)]
        return goals

class MDP(iMDP):
    """
    MDP object, inherits everything from iMDP but redefines probabilities in a frequentist manner
    """

    def samples_to_prob(self, N, N_in):
        """
        Turns samples to probabilities by dividing samples in a region by total number of samples
        """
        probs = [[0,1] for state in self.States]
        probs.append([0,1])
        for j, N_in_j in enumerate(N_in):
            if N_in_j > 0:
                freq_prob = N_in_j/N
                probs[j][0] = max(0, freq_prob - 0.01)
                probs[j][1] = min(1, freq_prob + 0.01)
            else:
                probs[j][1] = 0
        return probs

class hybrid_iMDP:
    """
    Class for hybrid formulation, contains a list of iMDPS, one for each mode
    """
    
    def __init__(self, Cont_state_space, Dynamics, parts, _beta = 0.01):
        self.iMDPs = [iMDP(Cont_state_space, dyn, parts, _beta) for dyn in Dynamics.individual_systems]
        self.mode_probs = Dynamics.transition_matrix
        self.N_modes = Dynamics.N_modes

    def find_state_index(self, x):
        return self.iMDPs[0].find_state_index(x)

    def update_probs(self, N):
        for imdp in self.iMDPs:
            imdp.update_probs(N) 
            # might be able to share probabilities if noises same, just need to think about checking actions enabled


class PRISM_writer:
    """
    Class for creating PRISM files
    """
    def __init__(self, _model, _N=-1, input_folder='input', output_folder='output', _explicit=False, _mode="interval", _horizon="infinite"):
        self.mode = _mode
        self.model = _model
        self.explicit = _explicit
        self.N = _N
        if _horizon != "infinite":
            self.horizon = str(N) + "_steps"
        else:
            self.horizon = _horizon
        input_prefix = input_folder + "/ScenAbs_"+_mode + "_" + _horizon 
        self.prism_filename = input_prefix + ".prism"
        self.spec_filename = input_prefix + ".pctl"
        
        self.state_filename = input_prefix + ".sta"
        self.label_filename = input_prefix + ".lab"
        self.transition_filename = input_prefix + ".tra"
        self.all_filename = input_prefix + ".all"

        file_prefix = output_folder + "/PRISM_out"
        self.vector_filename = file_prefix + "_vector.csv"
        self.policy_filename = file_prefix + "_policy.csv"

    def write(self):
        if self.explicit:
            self._write_explicit()
        else:
            self._write()

    def write_file(self, content, filename, mode="w"):
        """
        function for writing to a file
        """
        filehandle = open(filename, mode)
        filehandle.writelines(content)
        filehandle.close()

    def read(self):
        """
        Reads the results of solving the prism files from earlier
        """
        policy_file = self.policy_filename
        vector_file = self.vector_filename
        policy = np.genfromtxt(policy_file, delimiter=',', dtype='str')
        policy = np.flipud(policy)

        optimal_policy= np.zeros(np.shape(policy))
        optimal_delta= np.zeros(np.shape(policy))
        optimal_reward = np.zeros(np.shape(policy)[1])

        optimal_reward = np.genfromtxt(vector_file).flatten()
        for i, row in enumerate(policy):
            for j, value in enumerate(row):
                if value != '':
                    value_split = value.split('_')
                    optimal_policy[i,j] = int(value_split[1])
                    optimal_delta[i,j] = int(value_split[3])
                else:
                    optimal_policy[i,j] = -1
                    optimal_delta[i,j] = -1
        return optimal_policy, optimal_delta, optimal_reward

    def solve_PRISM(self,java_memory=2, prism_folder="~/Downloads/prism-imc/prism"):
        """
        function for solving iMDP using PRISM
        """
        import subprocess
        spec = self.specification

        options = ' -ex -exportadv "'+self.policy_filename+'"' + \
                  ' -exportvector "'+self.vector_filename+'"'

        if self.explicit:
            model_file = '"'+self.all_filename+'"'
            command = prism_folder+"/bin/prism -javamaxmem "+ \
                str(java_memory)+"g -importmodel "+model_file+" -pf '"+ \
                spec+"' "+options
        else:
            prism_file = self.prism_filename
            model_file = '"'+prism_file+'"'

            command = prism_folder + "/bin/prism -javamaxmem " + \
                    str(java_memory) + "g "+model_file+" -pf '"+spec+"' "+options
        subprocess.Popen(command, shell=True).wait()


class hybrid_PRISM_writer(PRISM_writer):
    """
    Class for dealing with hybrid formulations in PRISM
    """
    
    def _write_explicit(self):
        state_list = ['(x,m)']
        counter=0
        for m_num, m in enumerate(self.model.iMDPs):
            state_list += [str(counter)+':(-1,'+str(m_num)+')']
            state_list += [str(counter+i+1)+':('+str(i)+','+str(m_num)+')'\
                             for i in range(len(m.States))]
            counter += len(m.States)+1
        
        state_string = '\n'.join(state_list)
        
        self.write_file(state_string, self.state_filename)
        
        label_file_list = ['0="init" 1="deadlock" 2="reached"']

        counter = 0
        for m_num, m in enumerate(self.model.iMDPs):
            substring = [str(counter)+': 1']
            substring += ['' for i in m.States]
            for i, s in enumerate(m.States):
                substring[i+1] = str(counter+i+1)+': 0'

                if len(m.Actions_forward[i]) == 0:
                    substring[i+1] += ' 1'
            
            for i in m.Goals:
                substring[i+1] += ' 2'

            label_file_list += substring
            counter += len(m.States)+1

        label_file_string = '\n'.join(label_file_list)

        self.write_file(label_file_string, self.label_filename)
        
        nr_choices_absolute = 0
        nr_transitions_absolute = 0
        transition_file_list = []
        counter = 0
        for m_num, m in enumerate(self.model.iMDPs):
            transition_file_list_states = ['' for i in m.States]
            for i, s in enumerate(tqdm(m.States)):
                choice = 0
                selfloop = False
                #import pdb; pdb.set_trace()
                if len(m.Actions_forward[i]) > 0:
                    subsubstring = ['' for j in m.Actions_forward[i]]

                    for a_idx, a in enumerate(m.Actions_forward[i]):
                        action_label = "a_"+str(a)+"_d_"+str(m.dyn.grouped_timesteps)+"_m_"+str(m_num)
                        count_inn = 0
                        subsublist = []
                        for m_num_inn, m_inn in enumerate(self.model.iMDPs):
                            substring_start = str(counter+i+1) + ' '+str(choice)
                            if self.mode == "interval":
                                mode_prob = self.model.mode_probs[m_num, m_num_inn]
                                if mode_prob !=  0.0:
                                    interval_idxs = [j for j in m.trans_ids[a]]
                                    interval_strings = ["[" + str(prob[0]*mode_prob)
                                                        +","+str(prob[1]*mode_prob)+"]" for prob in m.trans_probs[a]]
                                    deadlock_string = interval_strings.pop(-1)
                                    subsubstring_a = [substring_start+' ' + str(count_inn) + ' ' \
                                                      +deadlock_string+' '+action_label]
                                    subsubstring_b = [substring_start+" "+str(count_inn+j+1)+" "+intv+" "+action_label
                                                        for (j, intv) in zip(interval_idxs, interval_strings)]
                                    subsublist += subsubstring_a + subsubstring_b
                                    count_inn += len(m.States)+1
                            else:
                                raise NotImplementedError

                        choice += 1
                        nr_choices_absolute += 1
                        nr_transitions_absolute += len(subsublist)

                        subsubstring[a_idx] = '\n'.join(subsublist)
                else:
                    if not selfloop:
                        if self.mode == 'interval':
                            selfloop_prob='[1.0,1.0]'
                        else:
                            selfloop_prob= '1.0'
                        subsubstring = [str(counter+i+1) + ' 0 '+str(counter+i+1)+' '+selfloop_prob]

                        selfloop = True

                        nr_choices_absolute += 1
                        choice += 1
                        nr_transitions_absolute += 1
                    else:
                        subsubstring = []
                substring = [subsubstring]
                transition_file_list_states[i] = substring
            transition_file_list += transition_file_list_states
            counter += len(m.States)+1

        flatten = lambda t: [item for sublist in t
                                  for subsublist in sublist
                                  for item in subsublist]
        transition_file_list = '\n'.join(flatten(transition_file_list))
        size_states = counter
        size_choices = nr_choices_absolute + 1
        size_transitions = nr_transitions_absolute + 1

        model_size = {'States': size_states,
                      'Choices': size_choices,
                      'Transitions':size_transitions}
        header = str(size_states)+' '+str(size_choices)+' '+str(size_transitions)+'\n'

        if self.mode == 'interval':
            firstrow = '0 0 0 [1.0,1.0]\n'
        else:
            firstrow = '0 0 0 1.0\n'

        self.write_file(header+firstrow+transition_file_list, self.transition_filename)

        self.specification = self.writePRISM_specification()

    def _write(self):
        """
        Writes .prism and .pctl file for abstraction, then stores the filenames
        """
        N = self.N
        model = self.model
        horizon = self.horizon
        mode = self.mode
        if horizon != "infinite":
            horizon = str(N) + "_steps"

        if horizon == "infinite":
            min_delta = N
            modeltype = type(model).__name__
            header = [
                "// "+modeltype+" (filter-based abstraction method) \n",
                "// Infinite horizon version \n\n"
                "mdp \n\n",
                # "const int xInit; \n\n",
                "const int regions = "+str(int(len(model.iMDPs[0].States)-1))+"; \n\n",
                "const int modes = "+str(int(model.N_modes-1))+"; \n\n",
                "module "+modeltype+"_abstraction \n\n",
                ]

            # Define variables in module
            variable_defs = [
                "\tx : [-1..regions]; \n\n",
                "\tm : [0..modes]; \n\n",   
                ]
        else:
            min_delta = model.iMDPs[0].dyn.grouped_timesteps
            header = [
                    "// " + type(model).__name__ + " (scenario-based abstraction method) \n\n",
                    "mdp \n\n",
                    "const int Nhor = " + str(int(N/model.dyn.grouped_timesteps)) + "; \n",
                    "const int modes = "+str(int(model.N_modes-1))+"; \n\n",
                    "const int regions = " + str(int(len(model.iMDPs[0].States)-1)) + "; \n\n", #maybe -1?
                    "module iMDP \n\n",
                    ]
            variable_defs = [
                    "\tk : [0..Nhor]; \n",
                    "\tx : [-1..regions]; \n\n",
                    ]
        self.write_file(header+variable_defs, self.prism_filename)
        delta = model.iMDPs[0].dyn.grouped_timesteps
        for m_num, m in enumerate(model.iMDPs):
            for k in range(0, N, min_delta):

                if horizon == 'finite':
                    action_defs += ["\t// Actions for k="+str(k)+"\n"]
                else:
                    action_defs = []

                action_defs += ["\t// Delta="+str(delta) + "\n"]

                bool_cont = k % delta == 0

                if (k + delta <= N and bool_cont) or horizon == 'infinite':

                    for a_num, a in enumerate(tqdm(m.Actions)):
                        actionLabel = "[a_"+str(a)+"_d_"+str(delta)+"_m_"+str(m_num)+"]"
                        enabledIn = m.Actions[a]

                        if len(enabledIn) > 0:
                            guardPieces = ["x="+str(state) for state in  enabledIn]
                            sep = " | "

                            if horizon == "infinite":
                                guardStates = sep.join(guardPieces)
                                guard = "m="+str(m_num)+" & ("+guardStates+")"
                                kprime = ""
                            else:
                                guardStates = sep.join(guardPieces)
                                guard = "m="+str(m_num)+"k="+str(int(k/min_delta)) + " & ("+guardStates+")"
                                kprime = "&(k'=k+"+str(1) + ")"

                            if mode == "interval":
                                succPieces=[]
                                for m_num_inner, _ in enumerate(model.iMDPs):
                                    mode_prob = model.mode_probs[m_num,m_num_inner]
                                    if mode_prob !=  0.0:
                                        interval_idxs = [str(i) for i in m.trans_ids[a]]
                                        interval_idxs[-1] = "-1"
                                        interval_strings = ["[" + str(prob[0]*mode_prob)
                                                            +","+str(prob[1]*mode_prob)+"]" for prob in m.trans_probs[a]]
                                        succPieces += [intv +" : (x'="+str(i)+")"+"&(m'="+str(m_num_inner)+")"
                                                      for (i,intv) in zip(interval_idxs, interval_strings)]
                            else:
                                raise NotImplementedError

                            sep = " + "
                            successors = sep.join(succPieces)

                            action_defs += "\t"+actionLabel+" " + guard + \
                                           " -> " + successors + "; \n"

                action_defs += ["\n\n"]
                self.write_file(action_defs, self.prism_filename, "a")

        if horizon == "infinite":
            footer = [
                "endmodule \n\n",
                "init x > -1 endinit \n\n"
                ]
        else:
            footer = [
                "endmodule \n\n",
                "init k=0 endinit \n\n"
                ]

        labelPieces = ["(x="+str(x)+")"\
                       for x in model.iMDPs[0].Goals] #assume shared goal
        sep = "|"
        labelGuard = sep.join(labelPieces)
        labels = [
            "// Labels \n",
            "label \"reached\" = " + labelGuard+"; \n"
            ]

        self.write_file(footer + labels, self.prism_filename, "a")

        self.specification = self.writePRISM_specification()

        print("Succesfully exported PRISM file")

    def writePRISM_specification(self):
        """
        Writes PRISM specification file in PCTL
        """
        N = self.N
        mode = self.mode
        model = self.model
        horizon = self.horizon
        if horizon == "infinite":
            horizonLen = int(N/model.iMDPs[0].dyn.grouped_timesteps)
            if mode == "estimate":
                specification = "Pmax=? [ F<="+str(horizonLen)+' "reached" ]'
            else:
                specification = "Pmaxmin=? [ F<="+str(horizonLen)+' "reached" ]'
        else:
            if mode == "estimate":
                specification = 'Pmax=? [ F "reached" ]'
            else:
                specification = 'Pmaxmin=? [ F "reached" ]'
        self.write_file(specification, self.spec_filename)
        return specification
