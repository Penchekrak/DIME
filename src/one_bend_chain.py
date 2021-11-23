from collections import OrderedDict


def get_state_dict_from_chain(t, state_dict_at_w1, state_dict_at_w2, state_dict_at_theta):
    state_dict = OrderedDict()
    keys = state_dict_at_theta.keys()
    for key in keys:
        if t < 0.5:
            state_dict[key] = state_dict_at_w1[key] + 2 * t * (state_dict_at_theta[key] - state_dict_at_w1[key])
        else:
            state_dict[key] = state_dict_at_theta[key] + 2 * (t - 0.5) * (state_dict_at_w2[key] - state_dict_at_theta[key])
    return state_dict