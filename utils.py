import torch
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import numpy as np
device = ('cpu')
import os
import scipy.ndimage
from scipy import ndimage
#from chamferdist import ChamferDistance
import ot
from scipy.spatial.distance import cdist
           
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
    '''
    binput_coords = binput_coords.to(device)
    binput_feats = binput_feats.to(device)
    btarget_coords = btarget_coords.to(device)
    btarget_feats = btarget_feats.to(device)
    bgrid_masks = bgrid_masks.to(device)
    #print('batched', btarget_coords[0], 'not batched', target_coords_l[0])
    #print('at batch', btarget_coords, len(btarget_coords), btarget_feats, len(btarget_feats))
    #print('len for l', target_coords_l, len(target_coords_l))
    '''

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

# accuracy metrics:

#chamferdist_func = ChamferDistance()


# Function to calculate Chamfer Distance
def chamfer_distance(track1, track2):
    dist1 = np.mean(np.min(cdist(track1, track2), axis=1))
    dist2 = np.mean(np.min(cdist(track2, track1), axis=1))
    return dist1 + dist2

# Function to calculate Earth Mover's Distance (EMD) using POT
def earth_movers_distance(track1, track2):
    cost_matrix = cdist(track1, track2)
    a = np.ones((track1.shape[0],)) / track1.shape[0]
    b = np.ones((track2.shape[0],)) / track2.shape[0]
    emd = ot.emd2(a, b, cost_matrix)
    return emd


# Filter tracks by X and Y ranges and compute distances
def calculate_distances(pred_data, target_data, x_min=114, x_max=128, y_min=140, y_max=154):
    track_ids = np.unique(pred_data[:, 0])
    valid_track_ids = []
    chamfer_distances = []
    emd_distances = []
    CH_funcs = []

    for track_id in track_ids:
        pred_track = pred_data[pred_data[:, 0] == track_id]
        target_track = target_data[target_data[:, 0] == track_id]

        # Filter by X or Y range for pred and target tracks
        pred_track_filtered = pred_track[((pred_track[:, 1] >= x_min) & (pred_track[:, 1] <= x_max)) |
                                         ((pred_track[:, 2] >= y_min) & (pred_track[:, 2] <= y_max))][:, 1:4]
        target_track_filtered = target_track[((target_track[:, 1] >= x_min) & (target_track[:, 1] <= x_max)) |
                                             ((target_track[:, 2] >= y_min) & (target_track[:, 2] <= y_max))][:, 1:4]

        if pred_track_filtered.size > 0 and target_track_filtered.size > 0:
            # print(pred_track_filtered)
            chamfer_dist = chamfer_distance(pred_track_filtered[:, 1:], target_track_filtered[:, 1:]) # Remove track ID, keep only XYZ
            emd_dist = earth_movers_distance(pred_track_filtered[:, 1:], target_track_filtered[:, 1:])
            # chamfer_dist_from_CH = chamfer_distance_from_chamferdist(pred_track_filtered, target_track_filtered)
            valid_track_ids.append(track_id)
            chamfer_distances.append(chamfer_dist)
            # print(type(chamfer_dist))
            emd_distances.append(emd_dist)
            # print(type(chamfer_dist_from_CH))
            # CH_funcs.append(chamfer_dist_from_CH)

    avg_chamfer_distance = np.mean(chamfer_distances)
    avg_emd_distance = np.mean(emd_distances)

    return valid_track_ids, avg_chamfer_distance, avg_emd_distance
