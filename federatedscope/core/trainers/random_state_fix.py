import torch
import numpy as np
import random

def save_random_state():
    """Save the current random state of PyTorch, NumPy and Python"""
    state = {
        'torch_state': torch.get_rng_state(),
        'numpy_state': np.random.get_state(),
        'python_state': random.getstate()
    }
    if torch.cuda.is_available():
        state['cuda_state'] = torch.cuda.get_rng_state_all()
    
    return state

def restore_random_state(state):
    """Restore the random state of PyTorch, NumPy and Python"""
    torch.set_rng_state(state['torch_state'])
    np.random.set_state(state['numpy_state'])
    random.setstate(state['python_state'])
    if torch.cuda.is_available() and 'cuda_state' in state:
        torch.cuda.set_rng_state_all(state['cuda_state'])
