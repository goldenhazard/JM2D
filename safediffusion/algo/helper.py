"""
Implementation of pieces of trajectory
"""

import numpy as np
import torch
from scipy.interpolate import interp1d

from safediffusion.algo.plan import ReferenceTrajectory

def match_trajectories(T_des, *args):
    """
    Given desired sample times T_des and any number of time vectors (1D arrays)
    and associated trajectories (2D arrays), reinterpolate the given trajectories
    linearly at the desired times.

    If T_des[0] < T_i[0], then the output Z_i is pre-padded with Z_i[:,0],
    and similarly if T_des[-1] > T_i[-1].

    Parameters:
        T_des (1D array): Desired sample times.
        *args: Variable length argument list where pairs of time vectors and 
               trajectories are expected, followed optionally by the interpolation type.

    Returns:
        List of interpolated trajectories.
    """
    if isinstance(args[-1], str):
        interp_type = args[-1]
        args = args[:-1]
    else:
        interp_type = 'linear'
        
    if np.isscalar(T_des):
        T_des = np.array([T_des])
    
    results = []
    for i in range(0, len(args), 2):
        T = np.array(args[i])
        Z = np.array(args[i+1])
        
        # If T_des exceeds the bounds of T, pad T and Z accordingly
        if T_des[0] < T[0]:
            T = np.insert(T, 0, T_des[0])
            Z = np.insert(Z, 0, Z[:, 0], axis=1)
        
        if T_des[-1] > T[-1]:
            T = np.append(T, T_des[-1])
            Z = np.append(Z, Z[:, -1].reshape(-1, 1), axis=1)
        
        if len(T) == 1 and np.array_equal(T_des, T):
            results.append(Z)
        else:
            interpolator = interp1d(T, Z, kind=interp_type, axis=1, fill_value="extrapolate")
            Z_des = interpolator(T_des)
            results.append(Z_des)
    
    return results

# def traj_uniform_acc(t, x0, v0, a):
#     """
#     Implementation of trajectory with uniform acceleration

#     Args:
#         t : (N_t,), time
#         x0: (B, D_x,), initial state
#         v0: (B, D_x,), initial velocity
#         a : (B, D_x,), acceleration
    
#     Returns:
#         x_des: (B, N_t, D_x), desired trajectory
#     """
#     assert t.ndim == 1 and x0.ndim == 2 and v0.ndim == 2 and a.ndim == 2

#     B, D_x = x0.shape
#     N_t = t.shape[0]

#     if not isinstance(t, torch.Tensor):
#         t = torch.tensor(t)
#     t_expanded = torch.unsqueeze(t, dim=1)
#     t_squared = t_expanded ** 2

#     x_vec = torch.unsqueeze(x0, axis=1) + torch.unsqueeze(v0, axis=1) * t_expanded + torch.unsqueeze(a, axis=1) * t_squared / 2
#     dx_vec = torch.unsqueeze(v0, axis=1) + torch.unsqueeze(a, axis=1) * t_expanded

#     return x_vec, dx_vec

def traj_uniform_acc(t, x0, v0, a):
    """
    Implementation of trajectory with uniform acceleration that supports arbitrary batch dimensions.

    Args:
        t :   (N_t,)         time
        x0:  (..., D_x)      initial state
        v0:  (..., D_x)      initial velocity
        a :  (..., D_x)      acceleration

    Returns:
        x_des:  (..., N_t, D_x), desired trajectory over time
        dx_des: (..., N_t, D_x), desired velocity over time
    """
    # Ensure t is a tensor and on the same device/dtype if needed
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
    else:
        t = t.to(dtype=x0.dtype, device=x0.device)

    *batch_dims, D_x = x0.shape
    N_t = t.shape[0]

    # Insert a time dimension before the D_x dimension
    # x0, v0, a: (..., D_x) -> (..., 1, D_x)
    x0_expanded = x0.unsqueeze(-2)  # (..., 1, D_x)
    v0_expanded = v0.unsqueeze(-2)  # (..., 1, D_x)
    a_expanded  = a.unsqueeze(-2)   # (..., 1, D_x)

    # Reshape t to align with the inserted time dimension
    # t: (N_t,) -> (1, N_t, 1)
    # This way, it can broadcast to match (..., 1, D_x) by expanding to (..., N_t, D_x)
    t_expanded = t.unsqueeze(0).unsqueeze(-1)  # (1, N_t, 1)

    # Compute position: x = x0 + v0*t + 0.5*a*t^2
    x_des = x0_expanded + v0_expanded * t_expanded + 0.5 * a_expanded * (t_expanded ** 2)
    # x_des: (..., 1, D_x) + (..., N_t, D_x) after broadcasting -> (..., N_t, D_x)

    # Compute velocity: dx = v0 + a*t
    dx_des = v0_expanded + a_expanded * t_expanded
    # dx_des: (..., N_t, D_x)

    return x_des, dx_des