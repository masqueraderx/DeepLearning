import torch
import numpy as np
from torch.autograd import Variable


data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor * tensor)
var_out = torch.mean(variable * variable)

var_out.backward()
print(variable.grad)
print(variable.data)