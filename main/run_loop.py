import main.iMDP as iMDP
import os

PRISM_MEM=12
def run(init_state, dyn, test_imdp,  grid, min_lb,init_samples=25, max_iters=20, max_samples=6401):
    lb_sat_prob = 0
    i = 0
    samples = init_samples
    init_id = test_imdp.find_state_index(init_state.T)
    while lb_sat_prob < min_lb and i < max_iters and samples < max_samples:
        print("Computing new probabilities with " + str(samples) + " samples")
        test_imdp.update_probs(samples)
        output_folder = 'output/'+type(dyn).__name__+'/'+str(samples)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder+'/')
        input_folder = 'input/'+type(dyn).__name__+'/'+str(samples)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder+'/')
        print("Writing PRISM files")
        writer = iMDP.PRISM_writer(test_imdp, dyn.horizon, input_folder, output_folder)
        writer.write()
        print("Solving iMDP")
        policy_file, vec_file = writer.solve_PRISM(PRISM_MEM)
        opt_pol, opt_delta, rew = writer.read()
        lb_sat_prob = rew[init_id]
        print("lower bound on initial state: "+lb_sat_prob)
        i+=1
        samples *= 2
    return opt_pol, opt_delta, rew

