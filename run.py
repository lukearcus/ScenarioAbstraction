import sys
import numpy as np
import UI.plot_funcs as plot_funcs
import UI.choices as opt
from main.model_defs import get_imdp
from main.run_loop import run
from Simulation.run import run as run_sim

start_samples = 100
num_samples=12801
iters=1

def main():
    """
    Main function for running Code
    """
    lb_sat_prob=0.5
    model = opt.model_choice()
    load_sel = opt.load_choice()
    save_sel = opt.save_choice()
    if model != "1room heating":
        noise_lvl = opt.noise_choice()
    else:
        noise_lvl = None
    imdp_abstr, ss, dyn, init_state, grid, model_name = get_imdp(load_sel, model, noise_lvl, save_sel)
    opt_pol, opt_rew = run(init_state, dyn, imdp_abstr, grid, lb_sat_prob, model_name, init_samples=start_samples, max_samples=num_samples, max_iters=iters)
    if model.split("_")[0] == "UAV":
        ax = ss.draw_space([0,1,2])
    else:
        ax=None
    plot_funcs.create_plots(model, opt_pol, opt_rew, imdp_abstr, init_state, ax)
    # draw some other nice things here
    import pdb; pdb.set_trace()
    run_sim(imdp_abstr, dyn, opt_pol)
    return 0

if __name__ == '__main__':
    sys.exit(main())
