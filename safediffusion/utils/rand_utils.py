import os
import torch
import numpy as np
import random

def set_random_seed(seed, deterministic_torch = True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # If you are using CUDA
    # Ensure deterministic behavior in PyTorch
    torch.use_deterministic_algorithms(deterministic_torch)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
        # Ensure that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = deterministic_torch
        torch.backends.cudnn.benchmark = False