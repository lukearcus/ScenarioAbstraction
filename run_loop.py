import iMDP
import os

PRISM_MEM=12
def run(init_state, dyn, ss,  grid, min_lb,init_samples=25, max_iters=20, max_samples=6401):
    lb_sat_prob = 0
    i = 0
    test_imdp = iMDP.iMDP(ss, dyn, grid, 0)
    samples = init_samples
    init_id = test_imdp.find_state_index(init_state.T)
    while lb_sat_prob < min_lb and i < max_iters and samples < max_samples:
        print("Computing new probabilities with " + str(samples) + " samples")
        test_imdp.update_probs(samples)
        output_folder = 'output/'+type(dyn).__name__+'/'+str(samples)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder+'/')
        writer = iMDP.PRISM_writer(test_imdp, dyn.horizon, output_folder)
        policy_file, vec_file = iMDP.solve_PRISM(writer.filename, writer.specification, output_folder, PRISM_MEM)
        opt_pol, opt_delta, rew = writer.read_results(policy_file, vec_file)
        lb_sat_prob = rew[init_id]
        i+=1
        samples *= 2
    return opt_pol, opt_delta, rew

