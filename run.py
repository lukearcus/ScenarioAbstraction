import sys
import numpy as np
import UI.plot_funcs as plot_funcs
import UI.choices as opt
from main.model_defs import get_imdp
from main.run_loop import run
from Simulation.run import run as run_sim

start_samples = 25
num_samples=12801
iters=1
sim=True

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
    if sim:
        if model.split("_")[0] == "UAV":
            ax = ss.draw_space([0,1,2])
        else:
            ax = ss.draw_space()
        sim_states = run_sim(imdp_abstr, dyn, opt_pol)
        sim_x = [float(elem[0]) for elem in sim_states]
        sim_y = [float(elem[1]) for elem in sim_states]
        sim_res = [sim_x, sim_y]
        if ss.n_dims > 2:
            sim_z = [float(elem[2]) for elem in sim_states]
            sim_res.append(sim_z)
    else:
        ax=None
        sim_res = None
    plot_funcs.create_plots(model, opt_pol, opt_rew, imdp_abstr, init_state, ax, sim_res)
    # draw some other nice things here

    return 0

if __name__ == '__main__':
    sys.exit(main())
