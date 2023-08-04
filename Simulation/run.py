from .controller import controller


def run(imdp, dyn, pol):
    control = controller(imdp.iMDPs, dyn, pol)
    states = []
    for t in range(dyn.horizon):
        states.append(dyn.state)
        c_act, d_act = control.get_action(dyn.state)
        dyn.state_update(c_act, d_act)
    return states
