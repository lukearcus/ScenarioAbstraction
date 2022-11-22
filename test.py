import sys
import numpy as np
import UI.plot_funcs as plot_funcs
from main.model_defs import get_imdp
from main.run_loop import run

start_samples = 100
num_samples=12801
iters=1

def main():
    """
    Main function for running Code
    """
    lb_sat_prob=0.5
    model = "steered_n_room_heating"
    load_sel = "N"
    save_sel = "N"
 
    if model != "1room heating":
        noise_lvl = 2
    else:
        noise_lvl = None
    imdp_abstr, ss, dyn, init_state, grid, model_name = get_imdp(load_sel, model, noise_lvl, save_sel,2)
    opt_pol, opt_rew = run(init_state, dyn, imdp_abstr, grid, lb_sat_prob, model_name, init_samples=start_samples, max_samples=num_samples, max_iters=iters)
    if model.split("_")[0] == "UAV":
        ax = ss.draw_space([0,1,2])
    else:
        ax=None
    plot_funcs.create_plots(model, opt_pol, opt_rew, imdp_abstr, init_state, ax)
    # draw some other nice things here
    import pdb; pdb.set_trace()
    return 0

if __name__ == '__main__':
    sys.exit(main())
