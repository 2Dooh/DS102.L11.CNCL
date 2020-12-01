import time
from utils.flops_benchmark import add_flops_counting_methods
import torch
import numpy as np

def print_info(self):
    n_params = (np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, self.model.parameters())) / 1e6)

    self.model = add_flops_counting_methods(self.model)
    self.model.eval()
    self.model.start_flops_count()
    random_data = torch.randn(1, *self.data_info['input_size'])
    self.model(torch.autograd.Variable(random_data).to(self.device))
    n_flops = np.round(self.model.compute_average_flops_cost() / 1e6, 4)

    print('{} Million of parameters | {} MFLOPS'.format(n_params, n_flops))