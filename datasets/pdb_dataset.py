import torch
import os
from torch.utils.data import Dataset
from .utils import PDBtoData # Adjusted for relative import if utils.py is in the same directory

class PDBDataset(Dataset):
    def __init__(self, file_list, data_config, dataset_name="train"):
        self.file_list = file_list
        self.data_config = data_config
        self.dataset_name = dataset_name
        print(f"Initialized {self.dataset_name} dataset with {len(self.file_list)} samples.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        pdb_path = self.file_list[idx]
        try:
            data = PDBtoData(
                pdb_path,
                num_posenc=self.data_config.num_posenc,
                num_rbf=self.data_config.num_rbf,
                knn_num=self.data_config.knn_num
            )
            return data
        except Exception as e:
            print(f"Error loading PDB file {pdb_path}: {e}")
            # Return None or a dummy data structure if you want to skip problematic files
            # For now, let re-raise to be aware of issues
            raise e

def custom_collate_fn(batch_list):
    if not batch_list:
        return None 
    
    # Filter out None items if __getitem__ can return None for problematic files
    batch_list = [item for item in batch_list if item is not None]
    if not batch_list:
        return None

    keys = batch_list[0].keys()
    collated_batch = {k: [dic[k] for dic in batch_list] for k in keys}
    
    # The model's forward pass calls geo_batch internally.
    # So the DataLoader should yield a dictionary in the format `collated_batch`.
    # The training loop will add `z_t` and `noise_level` to this dictionary later.
    return collated_batch
