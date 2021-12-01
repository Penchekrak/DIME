from collections import OrderedDict

import torch

class SmallNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = SmallNet()
st1 = SmallNet().state_dict()
st2 = SmallNet().state_dict()

state_dict_at_theta = OrderedDict()
theta_params = []
for param_key in st1.keys():
    state_dict_at_theta[param_key] = (st1[param_key] + st2[param_key]) / 2
    try:
        state_dict_at_theta[param_key].requires_grad_()
        theta_params.append(state_dict_at_theta[param_key])
    except RuntimeError:
        pass

def get_state_dict_from_chain(t, state_dict_at_w1, state_dict_at_w2, state_dict_at_theta):
    state_dict = OrderedDict()
    if t < 0.5:
        for param_key in state_dict_at_w1.keys():
            state_dict[param_key] = state_dict_at_w1[param_key] * (1-t*2) + \
                                    state_dict_at_theta[param_key] * (t*2)
    else:
        for param_key in state_dict_at_w1.keys():
            state_dict[param_key] = state_dict_at_w2[param_key] * (t*2-1) + \
                                    state_dict_at_theta[param_key] * (2-t*2)
    return state_dict

t = 0.25
state_dict = get_state_dict_from_chain(t, st1, st2, state_dict_at_theta)

missing_keys = []
unexpected_keys = []
error_msgs = []

# copy state_dict so _load_from_state_dict can modify it
metadata = getattr(state_dict, '_metadata', None)
def load(module, state_dict, prefix=''):
    module._load_from_state_dict(
        state_dict, prefix, metadata, True, missing_keys, unexpected_keys, error_msgs)
    for name, child in module._modules.items():
        if child is not None:
            load(child, state_dict, prefix + name + '.')

load(model, state_dict)
x = torch.tensor([[2.0, 3.0]])
loss = model(x)
loss.backward()
print(loss, state_dict)

for param in model.parameters():
    print(param, param.grad)
for param in theta_params:
    print(param, param.grad)
