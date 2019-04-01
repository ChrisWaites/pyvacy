import math
import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad

def make_optimizer_class(cls):

    class DPOptimizerClass(cls):

        def __init__(self, l2_norm_clip, noise_multiplier, num_minibatches, *args, **kwargs):
            """
            Args:
                l2_norm_clip (float): An upper bound on the 2-norm of the gradient w.r.t. the model parameters
                noise_multiplier (float): TBD
                num_minibatches (int): TBD
            """
            self.l2_norm_clip = l2_norm_clip
            self.noise = Normal(0.0, noise_multiplier * l2_norm_clip / num_minibatches)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

        def step(self, closure=None):
            # Calculate the 2-norm of the gradient w.r.t. all parameters in the model
            total_norm = torch.Tensor([0.0])
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        total_norm.add_(p.grad.data.norm().pow(2))
            total_norm = float(total_norm.sqrt())

            # Clip gradients given total norm and apply noise
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.div_(max(1, total_norm / self.l2_norm_clip))
                        p.grad.data.add_(self.noise.sample(p.grad.data.size()).to(self.device))

            super(DPOptimizerClass, self).step(closure)

    return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)

