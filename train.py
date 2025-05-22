import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np # Still needed for np.random.seed in set_random_seed if kept local, or other np uses
# import random # No longer needed if set_random_seed is from utils
import logging
import functools # Might not be strictly needed based on current plan
# import tree # Might not be strictly needed, geo_batch handles its data

from ml_collections import ConfigDict

# Project imports
from configs.train_ribodiffusion import get_config
from models.utils import create_model # Changed from GVPTransCond direct import
from models.ema import ExponentialMovingAverage
from diffusion.noise_schedule import NoiseScheduleVP
from datasets.pdb_dataset import PDBDataset, custom_collate_fn
from datasets import utils as du
from utils import set_random_seed, save_checkpoint, restore_checkpoint # Added
# from utils import get_data_scaler, get_data_inverse_scaler # Skipping for now
from run_lib import get_optimizer

# Constants
RNA_ALPHABET_SIZE = len(du.RNA_ALPHABET) # Should be 4 for A,C,G,U

# --- Main Training Function ---
def main(config: ConfigDict):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialization
    set_random_seed(config.seed)
    ckpt_dir = config.train.get("checkpoint_dir", "./checkpoints/ribodiffusion") # Get from config or default
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        logging.info(f"Created checkpoint directory: {ckpt_dir}")

    device = config.device
    logging.info(f"Using device: {device}")

    # Noise schedule
    noise_scheduler = NoiseScheduleVP(
        schedule=config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
        T=config.sampling.steps # Assuming T for training is related to sampling steps, or define separately
    )

    # Model
    # model = GVPTransCond(config.model).to(device) # Old way
    model = create_model(config) # New way, create_model handles device and DataParallel
    logging.info(f"Model initialized: {config.model.name}")
    # Note: create_model in models/utils.py wraps with DataParallel.
    # Access original model via model.module if needed for state_dict or specific methods.

    # EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    
    # Optimizer
    optimizer = get_optimizer(config, model.parameters()) # From run_lib
    # If get_optimizer includes scheduler logic based on config.optim.warmup, that's handled.
    # Otherwise, a simple AdamW is used. For more complex schedulers, add here.
    
    # Data Loading (Dummy file lists for now)
    # TODO: Replace with actual file list loading from config.train.dataset_path
    pdb_dir = "./example/" 
    if not os.path.exists(pdb_dir) or not os.listdir(pdb_dir):
        logging.warning(f"PDB directory {pdb_dir} is empty or does not exist. Using dummy PDB data.")
        # Create a dummy PDB file if it doesn't exist for the script to run
        dummy_pdb_path = os.path.join(pdb_dir, "R1107_dummy.pdb")
        if not os.path.exists(dummy_pdb_path):
            if not os.path.exists(pdb_dir): os.makedirs(pdb_dir)
            with open(dummy_pdb_path, "w") as f:
                f.write("ATOM      1  N   URA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
                f.write("ATOM      2  C1' URA A   1       0.000   0.000   1.458  1.00  0.00           C\n")
        file_list_train = [dummy_pdb_path]
        file_list_val = [dummy_pdb_path]
    else:
        all_pdbs = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
        if not all_pdbs:
            raise ValueError(f"No PDB files found in {pdb_dir}. Please provide data.")
        file_list_train = all_pdbs
        file_list_val = all_pdbs


    logging.info(f"Training with {len(file_list_train)} PDBs, Validating with {len(file_list_val)} PDBs.")

    train_dataset = PDBDataset(file_list_train, config.data, dataset_name="train")
    val_dataset = PDBDataset(file_list_val, config.data, dataset_name="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=config.train.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.train.batch_size, # Can be different for val
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=config.train.num_workers
    )

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # Training Loop
    start_epoch = 0
    # Try to restore checkpoint if a path is provided
    resume_ckpt_path = config.train.get("resume_checkpoint_path")
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        # model.module is for DataParallel, adjust if not using DP or if create_model changes
        model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        state_objects_to_load = {
            'model': model_to_load,
            'optimizer': optimizer,
            'ema': ema,
            'step': 0 # This will be updated by restore_checkpoint
        }
        loaded_state = restore_checkpoint(resume_ckpt_path, state_objects_to_load, device)
        start_epoch = loaded_state['step'] # restore_checkpoint from utils.py stores epoch/step in 'step'
        logging.info(f"Restored checkpoint from {resume_ckpt_path}. Starting at epoch {start_epoch}.")
    else:
        if resume_ckpt_path:
            logging.warning(f"Resume checkpoint path {resume_ckpt_path} not found. Starting from scratch.")
        start_epoch = 0

    logging.info(f"Starting training from epoch {start_epoch + 1}")
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(train_loader):
            if batch_data is None: # Skip if custom_collate_fn returned None
                logging.warning(f"Skipping step {step+1} in epoch {epoch+1} due to empty batch.")
                continue

            # Data preparation (as per revised plan)
            true_seq_num_list = [s.to(device) for s in batch_data['seq']] # list of [L] on device
            x_0_one_hot_list = [F.one_hot(s, num_classes=RNA_ALPHABET_SIZE).float() for s in true_seq_num_list] # list of [L, C] on device

            # Timesteps for each item in the batch
            t_for_batch = torch.randint(0, noise_scheduler.T, (len(x_0_one_hot_list),), device=device).long()
            
            alpha_t, sigma_t = noise_scheduler.marginal_prob(t_for_batch) # alpha_t, sigma_t are [B]

            epsilon_list = [torch.randn_like(x_0) for x_0 in x_0_one_hot_list] # list of [L,C] noise

            # Create x_t_list (noised one-hot sequences)
            x_t_list = []
            for i in range(len(x_0_one_hot_list)):
                x_0_item = x_0_one_hot_list[i] # [L, C]
                eps_item = epsilon_list[i]     # [L, C]
                # alpha_t[i] and sigma_t[i] are scalars, need to be reshaped for broadcasting
                noised_x_0 = alpha_t[i].view(1, 1) * x_0_item + sigma_t[i].view(1, 1) * eps_item
                x_t_list.append(noised_x_0)


            noise_levels_stacked = torch.log(alpha_t**2 / sigma_t**2) # [B]

            # Prepare model input batch
            # geo_batch (inside model) will handle moving items like 'coords', 'node_s', etc., to device
            # and converting lists to batched tensors.
            # 'seq' (numerical original) and 'z_t' (noised one-hot) are critical.
            model_input_batch = {}
            for key, value in batch_data.items():
                if key not in ['seq', 'z_t']: # These are handled specially
                     # geo_batch expects lists of tensors for these features
                    model_input_batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                elif key == 'seq': # Ensure original numerical sequences are on device for geo_batch
                    model_input_batch['seq'] = true_seq_num_list


            model_input_batch['z_t'] = x_t_list # list of [L,C] noised one-hot, on device

            # Forward pass
            # model.forward expects `batch` and `noise_level`. cond_x is optional.
            # `geo_batch` inside model.forward will process `model_input_batch`.
            # Specifically, `model_input_batch['z_t']` will be stacked by `geo_batch` to `[B, L, C]`
            # and `model_input_batch['seq']` will be stacked to `[B, L]`.
            predicted_x_0_logits = model(model_input_batch, noise_level=noise_levels_stacked, cond_x=None)
            
            # Target for loss
            target_x_0_batched = torch.stack(x_0_one_hot_list) # [B, L, C], already on device

            # Loss calculation
            loss = loss_fn(predicted_x_0_logits, target_x_0_batched)

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            if hasattr(config.optim, 'grad_clip') and config.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
            optimizer.step()

            # EMA update
            ema.update(model.parameters())

            epoch_loss += loss.item()

            if (step + 1) % config.train.print_freq == 0:
                lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch [{epoch+1}/{config.train.epochs}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {lr:.6f}")

        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        logging.info(f"End of Epoch {epoch+1}, Average Training Loss: {avg_epoch_loss:.4f}")

        # Validation Loop
        model.eval()
        ema.store(model.parameters()) # Store original params
        ema.copy_to(model.parameters()) # Copy EMA params to model for validation

        val_loss = 0.0
        with torch.no_grad():
            for val_step, val_batch_data in enumerate(val_loader):
                if val_batch_data is None: continue

                val_true_seq_num_list = [s.to(device) for s in val_batch_data['seq']]
                val_x_0_one_hot_list = [F.one_hot(s, num_classes=RNA_ALPHABET_SIZE).float() for s in val_true_seq_num_list]
                
                val_t_for_batch = torch.randint(0, noise_scheduler.T, (len(val_x_0_one_hot_list),), device=device).long()
                val_alpha_t, val_sigma_t = noise_scheduler.marginal_prob(val_t_for_batch)
                val_epsilon_list = [torch.randn_like(x_0) for x_0 in val_x_0_one_hot_list]
                
                val_x_t_list = []
                for i in range(len(val_x_0_one_hot_list)):
                    x_0_item = val_x_0_one_hot_list[i]
                    eps_item = val_epsilon_list[i]
                    noised_x_0 = val_alpha_t[i].view(1, 1) * x_0_item + val_sigma_t[i].view(1, 1) * eps_item
                    val_x_t_list.append(noised_x_0)

                val_noise_levels_stacked = torch.log(val_alpha_t**2 / val_sigma_t**2)

                val_model_input_batch = {}
                for key, value in val_batch_data.items():
                    if key not in ['seq', 'z_t']:
                        val_model_input_batch[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                    elif key == 'seq':
                        val_model_input_batch['seq'] = val_true_seq_num_list
                
                val_model_input_batch['z_t'] = val_x_t_list
                
                val_predicted_x_0_logits = model(val_model_input_batch, noise_level=val_noise_levels_stacked, cond_x=None)
                val_target_x_0_batched = torch.stack(val_x_0_one_hot_list)
                
                current_val_loss = loss_fn(val_predicted_x_0_logits, val_target_x_0_batched)
                val_loss += current_val_loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        logging.info(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        ema.restore(model.parameters()) # Restore original model parameters for next training epoch

        # Checkpointing
        if (epoch + 1) % config.train.save_checkpoint_freq == 0 or (epoch + 1) == config.train.epochs:
            ckpt_file_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            # model.module is for DataParallel, adjust if not using DP or if create_model changes
            model_state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            state_to_save = {
                'model': model_state_dict_to_save,
                'optimizer': optimizer.state_dict(),
                'ema': ema.state_dict() if ema else None,
                'step': epoch + 1, # Save current epoch as 'step'
                'config': config.to_dict() # Save config for reproducibility
            }
            save_checkpoint(ckpt_file_path, state_to_save)
            logging.info(f"Saved checkpoint to {ckpt_file_path}") # save_checkpoint in utils doesn't log
            
    logging.info("Training completed.")

# --- Entry Point ---
if __name__ == '__main__':
    config = get_config() # Load default training config
    main(config)
