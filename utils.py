import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import numpy as np
#device = ('cpu')
import os
import scipy.ndimage
from scipy import ndimage
           
class particle_dataset(Dataset):
    
    def __init__(self, input_coords, input_feats, target_coords, target_feats, grid_masks):
        '''
        explain
        '''
        self.input_coords = input_coords
        self.input_feats = input_feats
        self.target_coords = target_coords
        self.target_feats = target_feats
        self.grid_masks = grid_masks
        
    def __len__(self):
        return len(self.input_coords)
    
    def __getitem__(self, idx):
        
        return {
            'input_coords': self.input_coords[idx],
            'input_feats': self.input_feats[idx],
            'target_coords':self.target_coords[idx],
            'target_feats': self.target_feats[idx],
            'grid_masks': self.grid_masks[idx]
        }


def custom_collate_fn(batch):
    
    input_coords_l, input_feats_l, target_coords_l, target_feats_l, grid_masks_l = [], [], [], [], []
    
    for item in batch:
        input_coords_l.append(item['input_coords'])
        input_feats_l.append(item['input_feats'])
        target_coords_l.append(item['target_coords'])
        target_feats_l.append(item['target_feats'])
        grid_masks_l.append(item['grid_masks'])

    input_coords_lnp = np.concatenate(input_coords_l, axis=0)
    input_feats_lnp = np.concatenate(input_feats_l, axis=0)
    target_coords_lnp = np.concatenate(target_coords_l, axis=0)
    target_feats_lnp = np.concatenate(target_feats_l, axis=0)
    grid_masks_lnp = np.concatenate(grid_masks_l, axis=0)
    
    input_coords_lt = torch.tensor(input_coords_lnp, dtype=torch.int32)
    input_feats_lt = torch.tensor(input_feats_lnp, dtype=torch.float32)
    target_coords_lt = torch.tensor(target_coords_lnp, dtype=torch.int32)
    target_feats_lt = torch.tensor(target_feats_lnp, dtype=torch.float32)
    grid_masks_lt = torch.tensor(grid_masks_lnp, dtype=torch.bool)
    #print('HEREEE', target_feats_l)
    binput_coords, binput_feats = ME.utils.sparse_collate(coords=input_coords_l, feats=input_feats_l)
    btarget_coords, btarget_feats = ME.utils.sparse_collate(coords=target_coords_l, feats=target_feats_l)
    
    bgrid_masks = grid_masks_lt
    binput_feats = binput_feats.view(-1, 1)
    btarget_feats = btarget_feats.view(-1, 1)
    
    binput_coords = binput_coords.to(dtype=torch.float32).int()
    binput_feats = binput_feats.to(dtype=torch.float32)
    btarget_coords = btarget_coords.to(dtype=torch.float32).int()
    btarget_feats = btarget_feats.to(dtype=torch.float32)

    return binput_coords, binput_feats, btarget_coords, btarget_feats, bgrid_masks, target_coords_l


def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def save_predictions(base_dir, predictions_coords, predictions_energy, input_coords, input_feats, target_coords, target_energy, epoch):
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Save the npz files
    np.savez(os.path.join(base_dir, f'pred_coords_{epoch}.npz'), *predictions_coords)
    np.savez(os.path.join(base_dir, f'pred_energy_{epoch}.npz'), *predictions_energy)
    np.savez(os.path.join(base_dir, f'input_coords_{epoch}.npz'), *input_coords)
    np.savez(os.path.join(base_dir, f'input_feats_{epoch}.npz'), *input_feats)
    np.savez(os.path.join(base_dir, f'target_coords_{epoch}.npz'), *target_coords)
    np.savez(os.path.join(base_dir, f'target_feats_{epoch}.npz'), *target_energy)

''' 
def load_npz_data(file_path):
    with np.load(file_path) as data:
        return [data[f'arr_{i}'] for i in range(len(data.files))]
'''

def load_npz_data(file_path):
    with np.load(file_path) as data:
        return [data[f'arr_{i}'] for i in range(len(data.files))]

  
