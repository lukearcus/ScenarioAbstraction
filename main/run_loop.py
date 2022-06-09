import main.iMDP as iMDP
import os

PRISM_MEM=12
def run(init_state, dyn, test_imdp,  grid, min_lb, model, init_samples=25, max_iters=20, max_samples=6401):
    lb_sat_prob = 0
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
        init_id = test_imdp.find_state_index(init_state.T)+1
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
        writer.write()
        print("Solving iMDP")
        writer.solve_PRISM(PRISM_MEM)
        opt_pol, rew = writer.read()
        lb_sat_prob = rew[tuple(init_id)]
        print("lower bound on initial state: "+str(lb_sat_prob))
        i+=1
        #i=200 # so we only run 1 loop
        samples *= 2
    return opt_pol, rew

