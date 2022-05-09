import numpy as np
import itertools
from scipy.stats import beta
from scipy.spatial import Delaunay
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

class iMDP:
    """
    iMDP maker
    
    notes:
        -transition prob definitions are only defined in terms of the goal state, might be better to properly define as p(s_i,a,s_j), but then we have a v. large number
        -determining actions here overestimates allowed actions which will not work out well because of point above (if we had a state dependend transition prob it would be fine)


    """
    def __init__(self, Cont_state_space, Dynamics, parts, num_samples, _beta = 0.01):
        """
        Const_state_space is StateSpace object
        parts is a list/ tuple of partitions for each dimension
        """
        self.beta = _beta
        self.N_d = Cont_state_space.n_dims
        self.ss = Cont_state_space
        self.dyn = Dynamics
        min_pos = Cont_state_space.valid_range[0]
        self.part_size = (Cont_state_space.valid_range[1]-Cont_state_space.valid_range[0])/parts
        
        self.States, self.corner_array, self.corners_flat = self.create_partition(min_pos, parts)
        self.Goals = self.find_goals()
        self.Unsafes = self.find_unsafes()
        self.end_corners_flat = np.reshape(self.corner_array[:,[0,-1],:],(-1,self.N_d))
        #import pdb; pdb.set_trace()
        self.A_pinv = np.linalg.pinv(self.dyn.A)
        self.B_pinv = np.linalg.pinv(self.dyn.B_full_rank) 
        
        np_incs = np.stack(([0 for i in range(self.N_d)],self.part_size)).T
        self.Actions = self.determine_actions()
        self.trans_probs = self.determine_probs(num_samples)

   # def create_probs(self, N):
   #    """
   #    function for adding new samples/ changing probabilities
   #    """
   #    self.trans_probs = self.determine_probs(N)

    def create_partition(self, min_pos, parts):
        state_increments = []
        print("Setting up iMDP states")
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
        corners_flat = np.reshape(corner_array, (-1, self.N_d))
        return states, corner_array, corners_flat

    def create_table(self, N):
        print("Building lookup table")
        beta_bar = self.beta/(2*N)
        table = np.zeros((N+1, 2))
        table[N,0] = 0
        table[N,1] = round(1-beta.ppf(beta_bar, N-1, 1),5)
        for k in progressbar.progressbar(range(N)):
                id_in = (k-1) // 2
                table[k,0] = round(1 - beta.ppf(1-beta_bar, id_in+1, N-k),5)
                if k == 0:
                    table[k,1] = 1
                else:
                    id_out = id_in + 1
                    table[k,1] = round(1 - beta.ppf(beta_bar, id_out+1, N-k), 5 )
        return table

    def determine_probs(self, N):
        """
        Enumerates over actions to find the probability of arriving in the goal state associated with that action
        """
        self.lookup_table = self.create_table(N)
        print("Finding transition probabilities")
        enabled_actions = [act for act in self.Actions if len(self.Actions[act]) > 0]
        #with Pool() as p:
        #    results = tqdm(p.imap(partial(self.comp_bounds, N=N), enabled_actions), total=len(self.Actions))
        #    probs = list(tuple(results))
        probs = {act:[] for act in enabled_actions}
        for action in tqdm(enabled_actions):
            probs[action] = self.comp_bounds(action, N)
        return probs

    def comp_bounds(self, action, N):
        """
        compute probability bounds by adding noise to goal position and checking the resulting state
        """
        actions = {i : [] for i in self.States if i not in self.Unsafes}
        #import time
        init = self.dyn.state
        pos = action
        #tic = time.perf_counter()
        resulting_states = self.add_noise(action, N)
        #toc = time.perf_counter()
        inds = self.find_state_index(resulting_states) # this still needs to be faster...
        N_in = [inds.count(i) for i in range(len(self.States)+1)]
        #tac = time.perf_counter()
        #print(toc-tic, "____", tac-toc)
        self.dyn.state = init
        return self.samples_to_prob(N, N_in)

    def add_noise(self, action, N):
        p = len(action)
        if hasattr(self.dyn, 'mu'):
            resulting_states = np.tile(np.expand_dims(action,1),(1,N)).T +\
                               np.random.multivariate_normal(self.dyn.mu, self.dyn.sigma, (N))
        else:
            resulting_states = np.zeros((N,p))
            self.dyn.state = np.expand_dims(pos,1)
            for i in range(N):
                resulting_states[i,:] = (np.expand_dims(pos,1) + self.dyn.noise()).T

        return resulting_states

    def samples_to_prob(self, N, N_in):
        probs = [[0,1] for state in self.States]
        probs.append([0,1])
        for j, N_in_j in enumerate(N_in):
            k = N-N_in_j
            probs[j][0] = self.lookup_table[k,0]
            probs[j][1] = self.lookup_table[k,1]
            #if k == N:
            #    probs[j][0] = 0
            #    probs[j][1] = 1- beta.ppf(beta_bar, N-1, 1)
            #else:
            #    id_in = (k-1) // 2
            #    probs[j][0] = 1 - beta.ppf(1-beta_bar, id_in+1, N_in_j)
            #    if k == 0:
            #        probs[j][1] = 1
            #    else:
            #        id_out = id_in + 1
            #        probs[j][1] = 1 - beta.ppf(beta_bar, id_out+1, N_in_j) 
            if j == len(N_in) - 1:
                low = probs[j][0]
                upp = probs[j][1]
                probs[j][0] = 1 - upp
                probs[j][1] = 1 - low
            #if N_in_j < N:
            #    if N_in_j == 0:
            #        probs[j][0] = 0
            #    else:
            #        probs[j][0] = beta.ppf(beta_bar, N_in_j + 1, N-(N_in_j+1)+1) # should precompute these
            #    probs[j][1] = beta.ppf(1-beta_bar, N_in_j+1, N-(N_in_j+1)+1)
            #else:
            #    probs[j][0] = 1
        return probs

    def find_state_index(self, x):
        if len(x.shape) < 2:
            test_pos = x[np.newaxis, np.newaxis, :]
            num_ins = 1
        else:
            test_pos = x[:, np.newaxis, :]
            num_ins = x.shape[0]
        shape =  (num_ins,self.corner_array.shape[0],2,self.corner_array.shape[2])
        signs = np.sign(self.end_corners_flat - test_pos)
        reshaped = signs.reshape(shape)
        summed = reshaped.sum(2)
        abs_sum = np.abs(summed).sum(2)

        # might be able to just check >= and should be last state
        return [np.where(abs_sum[i,:]==0)[0][0] if not np.all(abs_sum[i,:]) else len(self.States) for i in range(num_ins) ]
        #if state_ind.shape[0] > 0:
        #    return np.random.choice(state_ind)
        #else:
        #    return len(self.States)
        #for state in self.States:
        #    if np.all(x >= state - self.part_size/2) and np.all(x <= state + self.part_size/2):
        #        return self.States.index(state)
        #return len(self.States)

    def corner_to_state(self, corner):
        np.where()

    def determine_actions(self):
        print("Determining actions")
        u = [[self.dyn.u_min[i], self.dyn.u_max[i]] for i in range(len(self.dyn.u_max))]
        x_inv_area = np.zeros((2**len(self.dyn.u_max), self.dyn.A.shape[0]))
        for i, u_elem in enumerate(itertools.product(*u)):
            list_elem = list(u_elem)
            x_inv_area[i,:] = (self.A_pinv @ (self.dyn.B @ np.array(list_elem) + self.dyn.Q)).flatten()
        corners_array = self.corners_flat

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

            parallelo2cube = np.linalg.inv(basis_vectors)
            x_inv_area_normalised = x_inv_area @ parallelo2cube
            
            predSet_originShift = -np.average(x_inv_area_normalised, axis=0)

            allRegionVertices = corners_array @ parallelo2cube - predSet_originShift
            # use basis vectors
        else:
            x_inv_hull = Delaunay(x_inv_area, qhull_options='QJ')
            allRegionVertices = corners_array
        if dim_equal:
            actions = {i : [] for i in self.States if i not in self.Unsafes}
            for act in tqdm(actions):
                actions[act] = self.inv_reachable(act, allRegionVertices, parallelo2cube, n)
        else:
            actions = {i : [] for i in self.States if i not in self.Unsafes}
            for act in tqdm(actions):
                state_counter = {i: 0 for i in self.States if i not in self.Unsafes}
                A_inv_d = self.A_pinv @ np.array(act)
                all_vertices = A_inv_d - corners_array
                in_hull = x_inv_hull.find_simplex(all_vertices) >= 0
                if np.any(in_hull):
                    for val in corners_array[in_hull]:
                        for state in self.Corners_to_states[tuple(val)]:
                            if state not in self.Unsafes:
                                state_counter[state]+=1
                                if state_counter[state] == 2**self.N_d:
                                    actions[act].append(state)
                state_ids = np.where(enabled_in)[0]
                actions[act] = [self.States[state_id] for state_id in state_ids if self.States[state_id] not in self.Unsafes]
        return actions
    
    def inv_reachable(self, act, allRegionVertices, parallelo2cube, n):
        A_inv_d = self.A_pinv @ np.array(act)
        all_vert_normed = (A_inv_d @ parallelo2cube) - allRegionVertices

        ## 
        poly_reshape = np.reshape(all_vert_normed, (len(self.States),n*(2**n)))
        enabled_in = np.maximum(np.max(poly_reshape, axis=1), -np.min(poly_reshape, axis=1)) <= 1.0
        state_ids = np.where(enabled_in)[0]
        return [self.States[state_id] for state_id in state_ids if self.States[state_id] not in self.Unsafes]
    
    def find_unsafes(self):
        """
        Finds all iMDP states that have a centre in unsafe region
        """
        unsafes = [state for state in self.States if not self.ss.check_safe(state)]
        return unsafes

    def check_unsafe(self, state): # this is gonna be a probem
        overlap = True
        for i in range(self.N_d):
            for unsafe in self.ss.unsafes:
                overlap = overlap and state[i] < unsafe[1][i]
                overlap = overlap and state[i]+self.part_size[i] > unsafe[0][i]
        return overlap

    def find_goals(self):
        """
        checks if centre of state is a goal region
        """
        goals = [state for state in self.States if self.ss.check_goal(state)]
        return goals

    def backward_states(self, curr):
        reachable_states = []
        for state in self.States:
            centre = state + self.part_size/2
            if self.is_reachable(curr, centre):
                reachable_states.append(state)
        return reachable_states
    
    def is_reachable(self, d_j, x):
        """
        Check if we can go from x to d_j
        """
        u = self.B_pinv @ (d_j- self.dyn.A_full_rank @ x)
        return self.dyn.is_valid(u)

class MDP(iMDP):

    def samples_to_prob(self, N, N_in):
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


class PRISM_writer:

    def __init__(self, model, N=-1, mode="interval"):
        if N == -1:
            horizon = "infinite"
        else:
            horizon = str(N) + "_steps"
        self.filename = "ScenAbs_" + type(model.dyn).__name__ + "_" + mode + "_" + horizon + ".prism"
        
        if horizon == "infinite":
            raise NotImplementedError
        else:
            header = [
                    "// " + type(model).__name__ + " (scenario-based abstraction method) \n\n",
                    "mdp \n\n",
                    "const int Nhor = " + str(int(N/model.dyn.grouped_timesteps)) + "; \n",
                    "const int regions = " + str(int(len(model.States))) + "; \n\n", #maybe -1?
                    "module iMDP \n\n",
                    ]
            variable_defs = [
                    "\tk : [0..Nhor]; \n",
                    "\tx : [-1..regions]; \n\n",
                    ]
        self.write_file(header+variable_defs, self.filename)
        
        for k in range(0, N, model.dyn.grouped_timesteps):

            if horizon == 'finite':
                action_defs += ["\t// Actions for k="+str(k)+"\n"]
            else:
                action_defs = []

            action_defs += ["\t// Delta="+str(model.dyn.grouped_timesteps) + "\n"]

            bool_cont = k % model.dyn.grouped_timesteps == 0

            if (k + model.dyn.grouped_timesteps <= N and bool_cont) or horizon == 'infinite':

                for a_num, a in enumerate(model.Actions):
                    actionLabel = "[a_"+str(a_num)+"_d_"+str(model.dyn.grouped_timesteps)+"]"
                    enabledIn = model.Actions[a]

                    if len(enabledIn) > 0:
                        guardPieces = ["x="+str(i) for i, _ in enumerate(enabledIn)]
                        sep = " | "
                    
                        if horizon == "infinite":
                            guard = sep.join(guardPieces)
                            kprime = ""
                        else:
                            guardStates = sep.join(guardPieces)
                            guard = "k="+str(int(k/model.dyn.grouped_timesteps)) + " & ("+guardStates+")"
                            kprime = "&(k'=k+"+str(1) + ")"

                        if mode == "interval":
                            interval_idxs = [str(i) for i, state in enumerate(model.States)] + ["-1"]
                            interval_strings = ["[" + str(prob[0])
                                                +","+str(prob[1])+"]" for prob in model.trans_probs[a]]
                            succPieces = [intv +" : (x'="+str(i)+")"+kprime
                                          for (i,intv) in zip(interval_idxs, interval_strings)]
                        else:
                            raise NotImplementedError

                        sep = " + "
                        successors = sep.join(succPieces)

                        action_defs += "\t"+actionLabel+" " + guard + \
                                       " -> " + successors + "; \n"
            
            action_defs += ["\n\n"]
            self.write_file(action_defs, self.filename, "a")
        
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

        labelPieces = ["(x="+str(x)+")" for x in model.Goals]
        sep = "|"
        labelGuard = sep.join(labelPieces)
        labels = [
            "// Labels \n",
            "label \"reached\" = " + labelGuard+"; \n"
            ]

        self.write_file(footer + labels, self.filename, "a")

        self.specfile, self.specification = self.writePRISM_specification(mode, horizon, N, model)

        print("Succesfully exported PRISM file")

    def writePRISM_specification(self, mode, horizon, N, model):

        if horizon == "infinite":
            horizonLen = int(N/model.dyn.grouped_timesteps)
            if mode == "estimate":
                specification = "Pmax=? [ F<="+str(horizonLen)+' "reached" ]'
            else:
                specification = "Pmaxmin=? [ F<="+str(horizonLen)+' "reached" ]'
        else:
            if mode == "estimate":
                specification = 'Pmax=? [ F "reached" ]'
            else:
                specification = 'Pmaxmin=? [ F "reached" ]'
        specfile = "ScenAbs_" + type(model.dyn).__name__ + "_" + mode + "_" + horizon + ".pctl"
        self.write_file(specification, specfile)

        return specfile, specification

    

    def write_file(self, content, filename, mode="w"):
        filehandle = open(filename, mode)
        filehandle.writelines(content)
        filehandle.close()

def solve_PRISM(prism_file, spec, java_memory=2, prism_folder="~/Downloads/prism-imc/prism"):
    import subprocess
    file_prefix = "PRISM_out"
    policy_file = file_prefix + "_policy.csv"
    vector_file = file_prefix + "_vector.csv"
    options = ' -ex -exportadv "'+policy_file+'"' + \
              ' -exportvector "'+vector_file+'"'

    model_file = '"'+prism_file+'"'
    command = prism_folder + "/bin/prism -javamaxmem " + \
              str(java_memory) + "g "+model_file+" -pf '"+spec+"' "+options
    subprocess.Popen(command, shell=True).wait()

    return policy_file, vector_file
import progressbar
