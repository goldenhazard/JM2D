"""
SafeDiffuser Code

Code largely adapted from https://github.com/Weixy21/SafeDiffuser
"""
from abc import ABC, abstractmethod

import torch
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
import torch.nn as nn

class InvarianceModule(ABC):
    def __init__(self, invariance_index):
        self.invariance_index = invariance_index
        self.n_optvar = len(invariance_index)

    @abstractmethod
    def __call__(self, t, x, xp1, projector):
        """
        Args:
            x: current noisy sample (x_k) (B, H, Ta+Td)
            xp1: next noisy sample (x_{k-1})
            projector: projector object
        """
        pass

    def class_k_func(self, x):
        return x


class InvarianceRoSModule(InvarianceModule):
    def __init__(self, invariance_index, robust_term):
        """
        Implementation of RoS Invariance Module 

        Args:
            invariance_index: index of invariance
            robust_term: robust term
        """
        super().__init__(invariance_index = invariance_index)
        self.robust_term = robust_term
    
    def __call__(self, t, x, xp1, projector):
        """
        Args:
            x: current noisy sample (x_k) (B, H, Ta+Td)
            xp1: next noisy sample (x_{k-1})

        NOTE:
            1. constraints
            2. state action dimension
        """
        # Parse static variables
        (B, H, D) = x.shape
        n_optvar    = self.n_optvar
        start_t = 1 if projector.skip_initial_state else 0
        Ntime = H - start_t

        # remove goal dimension
        ref = xp1 - x # (B, H, Ta+To)

        # Negative cons means safe
        Cons, Grad = projector.compute_statewise_safety_constraint(x, return_gradient=True)
        Ncons = Cons.shape[-1]

        Cons = -torch.tensor(Cons).to(x.device).reshape(-1, Ncons) # (B*H, Ncons)
        Grad = -torch.tensor(Grad).to(x.device).reshape(-1, Ncons, D) # (B*H, Ncons, 6)

        #########################################################
        # CBF construction
        #########################################################
        b   = Cons - self.robust_term # (B*H, Ncons)
        Lgb = Grad[..., self.invariance_index] # (B*H, Ncons, 2)
        Lfb = 0

        # constraint Lgb * u + Lfb + alpha(b) >= 0
        G = -Lgb # (B*H, Ncons, 2)
        h = Lfb + self.class_k_func(b) # (B*H, Ncons)

        # objective 1/2 * u^T Q u + q^T u
        Q = Variable(torch.eye(n_optvar))
        Q = Q.unsqueeze(0).expand(G.shape[0], n_optvar, n_optvar).to(G.device)
        q = -ref[..., start_t:, self.invariance_index] # (B*H, 2)
        q = q.reshape(-1, n_optvar) # (B*H*Ntime, 2)

        e = Variable(torch.Tensor())

        # For numerical stability
        Z = torch.amax(G.abs(), dim=(1,2)) # (B*H,)
        G = G/Z[:, None, None]
        h = h/Z[:, None]

        # solve QP
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e) # (B*Ntime, 2)
        out = out.reshape(B, Ntime, n_optvar) # (B, Ntime, 2)

        if torch.isnan(out).sum() > 0:
            print("Ill-define LgB")

        rt = xp1.clone()   
        rt[...,start_t:, self.invariance_index] = x[...,start_t:, self.invariance_index] + out
        
        return rt


class InvarianceReSModule(InvarianceModule):
    def __init__(self, invariance_index, robust_term, t_relax=5, sign=100):
        """
        Implementation of ReS Invariance Module (relaxed safety)

        Args:
            invariance_index: index of invariance
            robust_term: robust term
        """
        super().__init__(invariance_index = invariance_index)
        self.robust_term = robust_term
        self.t_relax = t_relax
        self.sign = sign
    
    def __call__(self, t, x, xp1, projector):
        # Parse static variables
        (B, H, D) = x.shape
        n_optvar    = self.n_optvar
        start_t = 1 if projector.skip_initial_state else 0
        Ntime = H - start_t

        sign = self.sign if t >= self.t_relax else 0

        # remove goal dimension
        ref = xp1 - x # (B, H, Ta+To)

        # Negative cons means safe
        Cons, Grad = projector.compute_statewise_safety_constraint(x, return_gradient=True)
        Ncons = Cons.shape[-1]

        Cons = -torch.tensor(Cons).to(x.device).reshape(-1, Ncons) # (B*H, Ncons)
        Grad = -torch.tensor(Grad).to(x.device).reshape(-1, Ncons, D) # (B*H, Ncons, 6)

        #########################################################
        # CBF construction
        #########################################################
        b   = Cons - self.robust_term # (B*H, Ncons)
        Lgb = Grad[..., self.invariance_index] # (B*H, Ncons, 2)
        Lfb = 0

        # constraint Lgb * u + Lfb + alpha(b) >= 0
        G = -Lgb # (B*H, Ncons, 2)
        G_relax = sign * torch.eye(Ncons).unsqueeze(0).expand(G.shape[0], Ncons, Ncons).to(G.device)
        G = torch.cat([G, G_relax], dim=-1)

        h = Lfb + self.class_k_func(b) # (B*H, Ncons)

        # objective 1/2 * u^T Q u + q^T u
        n_optvar = self.n_optvar + Ncons
        q = -ref[..., start_t:, self.invariance_index] # (B, H, 2)
        q = q.reshape(-1, self.n_optvar) # (B, 2)
        q_relax = torch.zeros((G.shape[0], Ncons)).to(G.device) # (B*H, Ncons)
        q = torch.cat([q, q_relax], dim=-1) # (B*H, 2+Ncons)

        Q = Variable(torch.eye(n_optvar))
        Q = Q.unsqueeze(0).expand(G.shape[0], n_optvar, n_optvar).to(G.device)

        e = Variable(torch.Tensor())

        # For numerical stability
        Z = torch.amax(G.abs(), dim=(1,2)) # (B*H,)
        G = G/Z[:, None, None]
        h = h/Z[:, None]

        # solve QP
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e) # (B*Ntime, n_optvar)
        out = out.reshape(B, Ntime, n_optvar) # (B, Ntime, n_optvar)

        if torch.isnan(out).sum() > 0:
            print("Ill-define LgB")

        rt = xp1.clone()   
        rt[...,start_t:, self.invariance_index] = x[...,start_t:, self.invariance_index] + out[..., :self.n_optvar]
        
        return rt
    
class InvarianceTVSModule(InvarianceModule):
    def __init__(self, invariance_index, robust_term, t_bias = 5):
        """
        Implementation of TVS Invariance Module 

        Args:
            invariance_index: index of invariance
            robust_term: robust term
            t_relax: time to relax
            sign: sign of the relaxed constraint
        """
        super().__init__(invariance_index = invariance_index)
        self.robust_term = robust_term
        self.t_bias = torch.tensor(t_bias)
    
    def __call__(self, t, x, xp1, projector):
        # Parse static variables
        (B, H, D) = x.shape
        n_optvar    = self.n_optvar
        start_t = 1 if projector.skip_initial_state else 0
        Ntime = H - start_t

        # remove goal dimension
        ref = xp1 - x # (B, H, Ta+To)

        # Negative cons means safe
        Cons, Grad = projector.compute_statewise_safety_constraint(x, return_gradient=True)
        Ncons = Cons.shape[-1]

        Cons = -torch.tensor(Cons).to(x.device).reshape(-1, Ncons) # (B*H, Ncons)
        Grad = -torch.tensor(Grad).to(x.device).reshape(-1, Ncons, D) # (B*H, Ncons, 6)

        #########################################################
        # CBF construction
        #########################################################
        b   = Cons - self.robust_term + 1 - nn.Sigmoid()(self.t_bias - t)# (B*H, Ncons)
        Lfb = nn.Sigmoid()(self.t_bias - t) * (1-nn.Sigmoid()(self.t_bias - t)) # (B*H, Ncons)
        Lgb = Grad[..., self.invariance_index] # (B*H, Ncons, 2)
        Lfb = 0

        # constraint Lgb * u + Lfb + alpha(b) >= 0
        G = -Lgb # (B*H, Ncons, 2)

        h = Lfb + self.class_k_func(b) # (B*H, Ncons)

        # objective 1/2 * u^T Q u + q^T u
        q = -ref[..., start_t:, self.invariance_index] # (B, H, 2)
        q = q.reshape(-1, n_optvar) # (B, 2)

        Q = Variable(torch.eye(n_optvar))
        Q = Q.unsqueeze(0).expand(G.shape[0], n_optvar, n_optvar).to(G.device)

        e = Variable(torch.Tensor())

        # For numerical stability
        Z = torch.amax(G.abs(), dim=(1,2)) # (B*H,)
        G = G/Z[:, None, None]
        h = h/Z[:, None]

        # solve QP
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e) # (B*Ntime, n_optvar)
        out = out.reshape(B, Ntime, n_optvar) # (B, Ntime, n_optvar)

        if torch.isnan(out).sum() > 0:
            print("Ill-define LgB")

        rt = xp1.clone()   
        rt[...,start_t:, self.invariance_index] = x[...,start_t:, self.invariance_index] + out
        
        return rt