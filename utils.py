import torch
import os
import numpy as np
from sampling import post_process


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        print(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=True)  # change strict to False?
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def get_data_scaler(config):
    """Data normalizer"""
    # not consider bias here
    centered = config.data.seq_centered

    def scale_fn(seq):
        if centered:
            seq = seq * 2. - 1.
        return seq

    return scale_fn


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""

    centered = config.data.seq_centered

    def inverse_scale_fn(seq):
        if centered:
            seq = (seq + 1.) / 2.
        return seq

    return inverse_scale_fn