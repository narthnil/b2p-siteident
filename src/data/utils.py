import numpy as np


def worker_init_fn(worker_id):
    """Set seed for dataloader"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)
