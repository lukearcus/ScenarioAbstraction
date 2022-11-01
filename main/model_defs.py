import numpy as np
import main.Dynamics as Dynamics
import main.StateSpace as StateSpace
import main.iMDP as iMDP
import UI.choices as opt
import pickle

def get_imdp(load_sel, model, noise_lvl, save_sel):
    """
    Either loads or creates a fresh iMDP abstraction based on chosen model
    Then saves based on users preference (if generating a fresh abstraction)
    """
    if model == "n_room_heating" or model == "steered_n_room_heating":
        nr_rooms = opt.rooms_choice()
        model_name = model+"_"+str(nr_rooms)
    else:
        model_name = model
    if load_sel == 'Y':
        try:
            with open("stored_abstractions/"+model_name + '_imdp.pkl', 'rb') as inp:
                imdp_abstr = pickle.load(inp)
        except(FileNotFoundError):
            print("Existing abstraction not found, proceeding to create new")
            load_sel = "N"
    if model.split("_")[0] == "UAV":
        init = np.array([[-14, 8, 106, 0, -2, 0]]).T
        T=1
        lb_acc = -4
        ub_acc = 4
        grid= (15,9,7,3,3,3)
        ss = StateSpace.ContStateSpace(6, ((-15, -9, 105) + tuple(-2.25 for i in range(3)), [15, 9, 119]+[2.25 for i in range(3)]),
                                       [
                                           ((-11,-1,105, -2.25, -2.25, -2.25),
                                            (-5,3,115, 2.25, 2.25, 2.25)), # hole 1

                                           ((-11,5,105, -2.25, -2.25, -2.25),
                                            (-5,9,115, 2.25, 2.25, 2.25)),
                                           ((-11, 5, 113, -2.25, -2.25, -2.25),
                                            (-5, 9, 117, 2.25, 2.25, 2.25)),
                                           ((-11, 3, 105, -2.25, -2.25, -2.25),
                                            (-5, 5, 107, 2.25, 2.25, 2.25)),
                                           ((-1, 1, 105, -2.25, -2.25, -2.25),
                                            (3, 9, 111, 2.25, 2.25, 2.25)), # hole 2

                                           ((-1, 1, 115, -2.25, -2.25, -2.25),
                                            (3, 9, 117, 2.25, 2.25, 2.25)),
                                           ((-1, 1, 111, -2.25, -2.25, -2.25),
                                            (3, 3, 115, 2.25, 2.25, 2.25)),
                                           ((-1, 7, 111, -2.25, -2.25, -2.25),
                                            (3, 9, 115, 2.25, 2.25, 2.25)),
                                           ((-1, -3, 105, -2.25, -2.25, -2.25),
                                            (3, 1, 119, 2.25, 2.25, 2.25)), # tower

                                           ((3, -3, 105, -2.25, -2.25, -2.25),
                                            (9, 1, 111, 2.25, 2.25, 2.25)), # wall between

                                           ((-11, -5, 105, -2.25, -2.25, -2.25),
                                            (-7, -1, 113, 2.25, 2.25, 2.25)), # long obs

                                           ((-1, -9, 105, -2.25, -2.25, -2.25),
                                            (3, -3, 107, 2.25, 2.25, 2.25)),
                                           ((-1, -9, 115, -2.25, -2.25, -2.25),
                                            (3, -3, 119, 2.25, 2.25, 2.25)), # overhang

                                           ((11, -9, 105, -2.25, -2.25, -2.25),
                                            (15, -5, 107, 2.25, 2.25, 2.25)), # small last

                                           ((9, 5, 105, -2.25, -2.25, -2.25),
                                            (15, 9, 113, 2.25, 2.25, 2.25)), # next to goal
                                       ],
                                       [((11,1,105, -2.25, -2.25, -2.25),(15,5,109, 2.25, 2.25, 2.25))])

        if model == "UAV_gauss":
            mu=0
            sigma=noise_lvl*0.075
            dyn = Dynamics.Full_Drone_gauss(init, T, ub_acc, lb_acc, mu, sigma)
        elif model == "UAV_speed_choice":
            dyn = Dynamics.steered_drone_speed(init, 0, T)
            init = [init, 0]
        elif model == "UAV_dir_choice":
            dyn = Dynamics.steered_drone_dir(init, 0, T)
            init = [init, 0]
        elif model == "UAV_var_noise":
            dyn = Dynamics.drone_var_noise(init, 0, T)
            init = [init, 0]
        else:
            wind_speed = noise_lvl*15
            dyn = Dynamics.Full_Drone_dryden(init, T, ub_acc, lb_acc, 5)
    if model=="1room heating":
        init = np.array([[19.8],[37]])
        dyn = Dynamics.heat_1_room(init)
        ss = StateSpace.ContStateSpace(2, ((19.1, 36), (22.9, 40)), [], [((20.9, 36), (21.1, 40)) ])
        grid = (19,20)
    if model=="n_room_heating":
        if nr_rooms == 2:
            sigma = noise_lvl*0.01
            init_state = np.array([[21,21]]).T
            init_mode=0
            dyn = Dynamics.multi_room_heating(init_state, init_mode, sigma=sigma)
            init = [init_state, init_mode]
            ss = StateSpace.ContStateSpace(nr_rooms, ((20, 20), (25, 25)), [], [((23, 23), (24, 24)) ])
            grid=(40,40)
        else:
            raise NotImplementedError
    if model=="steered_n_room_heating":
        if nr_rooms == 2:
            sigma = noise_lvl*0.01
            init_state = np.array([[21,21]]).T
            init_mode = 0
            dyn = Dynamics.steered_multi_room(init_state, init_mode, sigma=sigma)
            init = [init_state, init_mode]
            ss = StateSpace.ContStateSpace(nr_rooms, ((20, 20), (25, 25)), [], [((23, 23), (24, 24)) ])
            grid=(40,40)
        else:
            raise NotImplementedError
    if model == "unsteered_test":
        sigma = noise_lvl*0.01
        init_state = np.array([[-10,10]]).T
        init_mode = 0
        dyn = Dynamics.unsteered_test(init_state, init_mode, sigma=sigma)
        init = [init_state, init_mode]
        ss = StateSpace.ContStateSpace(2, ((-20, -20), (20, 20)), [((0,0),(1,1))], [((9, 9), (11, 11)) ])
        grid=(100,100)
    if model == "steered_test":
        sigma = noise_lvl*0.01
        init_state = np.array([[-10,10]]).T
        init_mode = 0
        dyn = Dynamics.steered_test(init_state, init_mode, sigma=sigma)
        init = [init_state, init_mode]
        ss = StateSpace.ContStateSpace(2, ((-20, -20), (20, 20)), [((0,0),(1,1))], [((9, 9), (11, 11)) ])
        grid=(100,100)
    if model == "conv_test":
        sigma = noise_lvl*0.01
        init = np.array([[-10,10]]).T
        dyn = Dynamics.conv_test(init, sigma=sigma)
        ss = StateSpace.ContStateSpace(2, ((-20, -20), (20, 20)), [((-5,-5),(1,1))], [((9, 9), (11, 11)) ])
        grid=(100,100)
    if model == "non_conv_test":
        sigma = noise_lvl*0.01
        init = np.array([[-10,10]]).T
        dyn = Dynamics.non_conv_test(init, sigma=sigma)
        ss = StateSpace.ContStateSpace(2, ((-20, -20), (20, 20)), [((-1,-1),(1,1))], [((9, 9), (11, 11)) ])
        grid=(100,100)
        init = [init, 0]
    if model == "steered_conv_test":
        sigma = noise_lvl*0.01
        init_state = np.array([[-10,10]]).T
        init_mode = 0
        init = [init_state, init_mode]
        dyn = Dynamics.steered_conv_test(init_state, init_mode, sigma=sigma)
        ss = StateSpace.ContStateSpace(2, ((-20, -20), (20, 20)), [((-5,-5),(1,1))], [((9, 9), (11, 11)) ])
        grid=(100,100)
        
    if load_sel == "N":
        if dyn.hybrid == False:
            init = [init, 0]
            dyn = Dynamics.single_hybrid(dyn) # can make single system into a hybrid with 1 discrete mode
        if dyn.steered == False:
            dyn = Dynamics.steered_MC(dyn)
        imdp_abstr = iMDP.hybrid_iMDP(ss,dyn,grid)
        if save_sel == 'Y':
            with open("stored_abstractions/"+model_name+'_imdp.pkl', 'wb') as outp:
                pickle.dump(imdp_abstr, outp, pickle.HIGHEST_PROTOCOL)
    return imdp_abstr, ss, dyn, init, grid, model_name
