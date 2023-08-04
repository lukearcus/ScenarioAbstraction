import main.iMDP as iMDP
import os
import numpy as np
import options

PRISM_MEM=options.prism_mem
PRISM_PATH=options.prism_path
def run(init_state, dyn, test_imdp,  grid, min_lb, model, init_samples=25, max_iters=20, max_samples=6401, test=False):
    lb_sat_prob = 0
    max_samples=max(init_samples+1,max_samples)
    i = 0
    samples = init_samples
    if dyn.hybrid:
        try:
            init_id = test_imdp.find_state_index(init_state[0].T)[0][0]+1
        except(TypeError):
            init_id = test_imdp.find_state_index(init_state[0].T+1e-5)[0][0]+1 #perturb slightly
        init_mode = init_state[1]
        init_id += init_mode*(len(test_imdp.iMDPs[0].States)+1)
    else:
        init_id = test_imdp.find_state_index(init_state.T)[0][0]+1
    while lb_sat_prob < min_lb and i < max_iters and samples < max_samples:
        print("Computing new probabilities with " + str(samples) + " samples")
        test_imdp.update_probs(samples)
        output_folder = 'output/'+model+'/'+str(samples)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder+'/')
        input_folder = 'input/'+model+'/'+str(samples)
        if not os.path.exists(input_folder):
            os.makedirs(input_folder+'/')
        print("Writing PRISM files")
        writer = iMDP.hybrid_PRISM_writer(
                                test_imdp, dyn.horizon, input_folder, output_folder, _explicit=True
                                )
        # for complex formula, first we do P>=0.7(safe until warm)
        if test:
            writer.opt_thresh = False
            writer.max = True
            writer.thresh = 0.9

        writer.write()
        print("Solving iMDP")
        writer.solve_PRISM(PRISM_MEM, PRISM_PATH)
        opt_pol, rew = writer.read()
        lb_sat_prob = rew[tuple(init_id)]
        print("lower bound on initial state: "+str(lb_sat_prob))
        i+=1
        samples *= 2

        #Below is to do more complex formulae
        if test:
            second_sats = np.copy(rew)

            writer.max = False
            writer.thresh = 0.5

            goals = np.array([[[20, 20],[25,21]]])
            #unsafes = np.array([[[]]])
            for m in test_imdp.iMDPs:
               m.ss.goals = goals
               m.Goals = m.find_goals()

            writer._write_labels()
            writer.spec = "until"
            writer.specification = writer.writePRISM_specification()
            
            writer.solve_PRISM(PRISM_MEM)
            opt_pol, rew = writer.read()

            writer.max = True
            writer.thresh = 0.6
           
            rew_1 = rew[:1601]
            rew_2 = rew[1601:]
            for m in test_imdp.iMDPs:
               m.Goals = (np.where(rew_1)[0]-1).tolist()

            writer._write_labels()
            writer.spec = "until"
            writer.N = 1
            writer.specification = writer.writePRISM_specification()

            writer.solve_PRISM(PRISM_MEM)
            opt_pol, rew = writer.read()

            final_rew = np.logical_and(rew, second_sats)

            import pdb; pdb.set_trace()
            rew = final_rew

    return opt_pol, rew

