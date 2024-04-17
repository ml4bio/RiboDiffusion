import torch
from tqdm import tqdm
import numpy as np
import random
from models import *
from utils import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets import utils as du
from torch_geometric.data import Batch
import logging
import pickle
import functools
import tree
import copy
import time


def set_random_seed(config):
    seed = config.seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def vpsde_inference(config, save_folder,
                    pdb_file='./example/R1107.pdb'):
    """Runs inference for RNA inverse design in a given dir."""
    # Create directory for eval_folder
    os.makedirs(save_folder, exist_ok=True)

    # Initialize model
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    print('model size: {:.1f}MB'.format(model_size))

    # Checkpoint name
    checkpoint_path = './ckpts/exp_inf.pth'

    # Load checkpoint
    state = restore_checkpoint(checkpoint_path, state, device=config.device)
    ema.copy_to(model.parameters())

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scalar and inverse scalar
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup new sampling function for multi-state
    test_sampling_fn = get_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)
    pdb2data = functools.partial(du.PDBtoData, num_posenc=config.data.num_posenc,
                                 num_rbf=config.data.num_rbf, knn_num=config.data.knn_num)

    fasta_dir = os.path.join(save_folder, 'fasta')
    os.makedirs(fasta_dir, exist_ok=True)

    # run inference on a single pdb file
    print('Start inference on a single pdb file')
    pdb_id = pdb_file.replace('.pdb', '')
    if '/' in pdb_id:
        pdb_id = pdb_id.split('/')[-1]
    struct_data = pdb2data(pdb_file)
    struct_data = tree.map_structure(lambda x:
                                     x.unsqueeze(0).repeat_interleave(config.eval.n_samples, dim=0).to(config.device),
                                     struct_data)
    samples = test_sampling_fn(model, struct_data)

    # save to fasta dir
    for i in range(len(samples)):
        du.sample_to_fasta(samples[i], pdb_id,
                           os.path.join(fasta_dir, pdb_id + '_' + str(i) + '.fasta'))

    recovery_ = samples.eq(struct_data['seq']).float().mean().item()
    print(f'{pdb_id}, recovery_rate {recovery_:.4f}')
