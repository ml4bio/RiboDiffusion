import unittest
import torch
import os
import sys

# Add project root to sys.path to allow for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from configs.train_ribodiffusion import get_config
from models.utils import create_model
from diffusion.noise_schedule import NoiseScheduleVP
from datasets.pdb_dataset import PDBDataset, custom_collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim # Though get_optimizer is used, this might be for type hints or other optimizers
import torch.nn.functional as F
from utils import set_random_seed # get_optimizer is not in utils
from run_lib import get_optimizer # Corrected import for get_optimizer

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.config = get_config()
        # Override some config for faster testing
        self.config.device = torch.device('cpu') # Force CPU for testing
        self.config.train.batch_size = 1
        self.config.train.num_workers = 0
        self.config.model.node_h_dim = (32, 8) # Smaller model hidden dims
        self.config.model.edge_h_dim = (32, 1) # Smaller model hidden dims
        self.config.model.num_layers = 1 # Fewer GVP layers
        self.config.model.num_trans_layer = 1 # Fewer Transformer layers
        self.config.model.trans.encoder_embed_dim = 32 # Smaller transformer embed dim
        self.config.model.trans.encoder_attention_heads = 2
        self.config.model.trans.encoder_ffn_embed_dim = 64
        # Ensure model.out_dim is present and correct (e.g., 4 for RNA alphabet size)
        self.config.model.out_dim = 4


        set_random_seed(self.config.seed)

        # Ensure the example PDB file exists for the test
        self.example_pdb = os.path.join(project_root, 'example/R1107.pdb')
        if not os.path.exists(self.example_pdb):
            # Attempt to create a dummy PDB if the specific example is missing
            # This makes the test more resilient if R1107.pdb is not available
            # but a general PDB loading test is still possible.
            dummy_example_dir = os.path.join(project_root, 'example')
            if not os.path.exists(dummy_example_dir):
                os.makedirs(dummy_example_dir)
            
            # Use the same name so the test logic doesn't need to change if it's just for loading
            # Or, use a different name and adjust self.example_pdb
            # For now, we'll stick to failing if R1107.pdb is specifically required by test logic beyond just loading.
            self.fail(f"Example PDB file not found at {self.example_pdb}. Test cannot run.")


        self.file_list_test = [self.example_pdb]

    def test_single_training_step(self):
        # 1. Initialize Dataset and DataLoader
        test_dataset = PDBDataset(
            file_list=self.file_list_test,
            data_config=self.config.data,
            dataset_name="test" # Added for clarity in PDBDataset init print
        )
        self.assertTrue(len(test_dataset) > 0, "Dataset should not be empty")

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False, # No need to shuffle for a single test item
            num_workers=self.config.train.num_workers,
            collate_fn=custom_collate_fn
        )

        # 2. Initialize Model, Optimizer, Noise Scheduler
        model = create_model(self.config) # create_model already handles .to(self.config.device)
        optimizer = get_optimizer(self.config, model.parameters()) 
        noise_scheduler = NoiseScheduleVP(
            schedule=self.config.sde.schedule,
            continuous_beta_0=self.config.sde.continuous_beta_0,
            continuous_beta_1=self.config.sde.continuous_beta_1,
            T=self.config.sampling.steps # Using sampling steps for T, as in train.py
        )
        loss_fn = torch.nn.MSELoss()

        # 3. Get a single batch
        try:
            batch_data = next(iter(test_loader))
        except StopIteration:
            self.fail("DataLoader produced no batches.")
        
        self.assertIsNotNone(batch_data, "Batch data is None")
        self.assertIn('seq', batch_data, "Batch data missing 'seq' key")


        # 4. Perform a single training step (simplified from train.py)
        model.train()
        optimizer.zero_grad()

        # Data preparation 
        # Convert numerical sequences to one-hot
        # Ensure seq data is actually present and is a list of tensors
        self.assertTrue(isinstance(batch_data['seq'], list), "'seq' should be a list of tensors.")
        self.assertTrue(all(torch.is_tensor(s) for s in batch_data['seq']), "All items in 'seq' should be tensors.")

        x_0_one_hot_list = [F.one_hot(s.to(self.config.device), num_classes=self.config.model.out_dim).float() for s in batch_data['seq']]
        
        t_item = torch.randint(0, noise_scheduler.T, (len(x_0_one_hot_list),), device=self.config.device).long()
        alpha_t, sigma_t = noise_scheduler.marginal_prob(t_item)
        
        stacked_x_0 = torch.stack(x_0_one_hot_list) # Now [B, L, C] on device
        epsilon = torch.randn_like(stacked_x_0)

        # Perform operations with batched tensors
        x_t_batched = alpha_t.view(-1, 1, 1) * stacked_x_0 + sigma_t.view(-1, 1, 1) * epsilon
        noise_levels_batched = torch.log(alpha_t**2 / sigma_t**2)

        # Prepare batch for model: z_t should be a list of tensors for geo_batch
        model_input_batch = {k: v for k, v in batch_data.items()} 
        model_input_batch['z_t'] = [x_t_batched[i].to(self.config.device) for i in range(x_t_batched.shape[0])]
        
        # Ensure other list data that geo_batch uses is on device
        # Also ensure 'seq' (numerical) is on device for geo_batch if it uses it directly
        for key_to_device in ['coords', 'node_s', 'node_v', 'edge_s', 'edge_v', 'edge_index', 'mask', 'seq']:
            if key_to_device in model_input_batch and isinstance(model_input_batch[key_to_device], list):
                model_input_batch[key_to_device] = [
                    item.to(self.config.device) if torch.is_tensor(item) else item 
                    for item in model_input_batch[key_to_device]
                ]
            elif key_to_device in model_input_batch and torch.is_tensor(model_input_batch[key_to_device]): # For 'seq' if it's already a tensor
                 model_input_batch[key_to_device] = model_input_batch[key_to_device].to(self.config.device)

        
        predicted_x_0_logits = model(
            model_input_batch, 
            noise_level=noise_levels_batched 
        ) 

        target_x_0_batched = stacked_x_0 
        loss = loss_fn(predicted_x_0_logits, target_x_0_batched)

        self.assertTrue(torch.is_tensor(loss), "Loss is not a tensor.")
        self.assertFalse(torch.isnan(loss), "Loss is NaN.")
        self.assertFalse(torch.isinf(loss), "Loss is Inf.")

        loss.backward()
        optimizer.step()

        print(f"Test single training step loss: {loss.item()}")

if __name__ == '__main__':
    unittest.main()
