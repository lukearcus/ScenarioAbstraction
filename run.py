import pickle
import sys
import numpy as np
import main.Dynamics as Dynamics
import main.StateSpace as StateSpace
import main.iMDP as iMDP
import UI.plot_funcs as plot_funcs
import UI.choices as opt
from main.run_loop import run


def get_imdp(load_sel, model):
    """
    Either loads or creates a fresh iMDP abstraction based on chosen model
    Then saves based on users preference (if generating a fresh abstraction)
    """
    if load_sel == 'Y':
        try:
            with open("stored_abstractions/"+model + '_imdp.pkl', 'rb') as inp:
                imdp_abstr = pickle.load(inp)
        except(FileNotFoundError):
            print("Existing abstraction not found, proceeding to create new")
            load_sel = "N"
    if load_sel == "N":
        save_sel = opt.save_choice()
        if model == "UAV_gauss" or model == "UAV_dryden":
            init_pos = np.ones((3,1))*1
            init_pos[2] = 6
            init_vel = np.zeros((3,1))
            init = np.concatenate((init_pos,init_vel))
            T=1
            lb_acc = -4
            ub_acc = 4
            grid= (15,9,7,3,3,3)
            ss = StateSpace.ContStateSpace(6, ((-15, -9, 5) + tuple(-2.25 for i in range(3)), [15, 9, 19]+[2.25 for i in range(3)]),
                                           [
                                               ((-11,-1,5, -2.25, -2.25, -2.25),(-5,3,15, 2.25, 2.25, 2.25)), # hole 1
                                               ((-11,5,5, -2.25, -2.25, -2.25),(-5,9,15, 2.25, 2.25, 2.25)),
                                               ((-11, 5, 13, -2.25, -2.25, -2.25),(-5, 9, 17, 2.25, 2.25, 2.25)),
                                               ((-11, 3, 5, -2.25, -2.25, -2.25),(-5, 5, 7, 2.25, 2.25, 2.25)),
                                               ((-1, 1, 5, -2.25, -2.25, -2.25),(3, 9, 11, 2.25, 2.25, 2.25)), # hole 2
                                               ((-1, 1, 15, -2.25, -2.25, -2.25),(3, 9, 17, 2.25, 2.25, 2.25)),
                                               ((-1, 1, 11, -2.25, -2.25, -2.25),(3, 3, 15, 2.25, 2.25, 2.25)),
                                               ((-1, 7, 11, -2.25, -2.25, -2.25),(3, 9, 15, 2.25, 2.25, 2.25)),
                                               ((-1, -3, 5, -2.25, -2.25, -2.25),(3, 1, 19, 2.25, 2.25, 2.25)), # tower
                                               ((3, -3, 5, -2.25, -2.25, -2.25),(9, 1, 11, 2.25, 2.25, 2.25)), # wall between
                                               ((-11, -5, 5, -2.25, -2.25, -2.25),(-7, -1, 13, 2.25, 2.25, 2.25)), # long obs
                                               ((-1, -9, 5, -2.25, -2.25, -2.25),(3, -3, 7, 2.25, 2.25, 2.25)),
                                               ((-1, -9, 15, -2.25, -2.25, -2.25),(3, -3, 19, 2.25, 2.25, 2.25)), # overhang
                                               ((11, -9, 5, -2.25, -2.25, -2.25),(15, -5, 7, 2.25, 2.25, 2.25)), # small last
                                               ((9, 5, 5, -2.25, -2.25, -2.25),(15, 9, 13, 2.25, 2.25, 2.25)), # next to goal
    
                                           ],
                                           [((11,1,5, -2.25, -2.25, -2.25),(15,5,9, 2.25, 2.25, 2.25))])
    
            if model == "UAV_gauss":
                mu=0
                sigma=0.5
                dyn = Dynamics.Full_Drone_gauss(init, T, ub_acc, lb_acc, mu, sigma)
            else:
                wind_speed = 5 # wind speed at 6m: 5 low, 15 medium, 30 high
                dyn = Dynamics.Full_Drone_dryden(init, T, ub_acc, lb_acc, 5)
    
        if model=="1room heating":
    
            init = np.array([[19.8],[37]])
            dyn = Dynamics.heat_1_room(init)
            ss = StateSpace.ContStateSpace(2, ((19.1, 36), (22.9, 40)), [], [((20.9, 36), (21.1, 40)) ])
            grid = (19,20)
    
        imdp_abstr = iMDP.iMDP(ss,dyn,grid)
        if save_sel == 'Y':
            with open("stored_abstractions/"+model+'_imdp.pkl', 'wb') as outp:
                pickle.dump(imdp_abstr, outp, pickle.HIGHEST_PROTOCOL)
        return imdp_abstr, ss, dyn 

def main():
    """
    Main function for running Code
    """
    lb_sat_prob=0.25
    model = opt.model_choice()
    load_sel = opt.load_choice()
    imdp_abstr, ss, dyn = get_imdp(load_sel, model)
    opt_pol, opt_delta, opt_rew = run(init, test, imdp_abstr,grid,lb_sat_prob)
    plot_funcs.create_plots(model, opt_pol, opt_rew)
    # draw some other nice things here
    return 0

if __name__ == '__main__':
    sys.exit(main())
