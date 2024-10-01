import os
import sys
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split
import torch.optim as optim
import wandb
import argparse
import copy 
import random

sys.path.insert(0, "./../" )

os.environ["WANDB__SERVICE_WAIT"] = "300"

from model_skull import *
from utils_skull import *

device = ('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train or resume training a model')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to the checkpoint file')
parser.add_argument('--wandb_run_id', type=str, default=None, help='W&B (wandb) run id for resuming training')
args = parser.parse_args()


# Define batch size, lr, num_epochs
batch_size = 40 
lr=1e-3 
num_epochs = 150*20 # set to something long; in preinciple I simply monitor the accuracy plots to see when I am overtraining 
print_every = 10
model_name = "muons_10_lr_twoinactive_newE"

# W&B API Key
os.environ["WANDB_API_KEY"] = "put in your WANDB_API_KEY or set it in the environment via terminal"

# Initialize W&B
if args.resume and args.wandb_run_id:
    wandb.init(project='project name', entity='your wandb id', id=args.wandb_run_id, resume="must")
else:
    wandb.init(project='project name', entity='your wandb id')

# Log hyperparameters for W&B
wandb.config.update({
    "batch_size": batch_size,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "model_name": model_name
}, allow_val_change=True)
    

input_coords_path = r'/path/to/input_coords_energies.npz'
input_feats_path = r'/path/to/input_feats_energies.npz'
target_coords_path = r'/path/to/target_coords_energies.npz'
target_feats_path = r'/path/to/target_feats_energies.npz'

input_coords = load_npz_data(input_coords_path)
print('loaded input coords')
input_feats = load_npz_data(input_feats_path)
print('loaded input feats')
target_coords = load_npz_data(target_coords_path)
print('loaded target coords')
target_feats = load_npz_data(target_feats_path)
print('loaded target feats')
# Generate a new list of arrays with the same shapes as target_feats, filled with random True or False
# grid masks are a legacy thing that will be deleted later
grid_masks = [np.random.choice([True, False], size=arr.shape) for arr in target_feats]
print('loaded grid masks')

# Split the data into training and initial test sets
train_incoords, test_incoords, train_infeats, test_infeats, train_tgcoords, test_tgcoords, train_tgfeats, test_tgfeats, train_masks, test_masks = train_test_split(
    input_coords, input_feats, target_coords, target_feats, grid_masks, test_size=0.2, random_state=42
)

# Split the initial test set into final test and validation sets
test_incoords, val_incoords, test_infeats, val_infeats, test_tgcoords, val_tgcoords, test_tgfeats, val_tgfeats, test_masks, val_masks = train_test_split(
    test_incoords, test_infeats, test_tgcoords, test_tgfeats, test_masks, test_size=0.5, random_state=42
)

# Create datasets
train_dataset = particle_dataset(train_incoords, train_infeats, train_tgcoords, train_tgfeats, train_masks)
test_dataset = particle_dataset(test_incoords, test_infeats, test_tgcoords, test_tgfeats, test_masks)
val_dataset = particle_dataset(val_incoords, val_infeats, val_tgcoords, val_tgfeats, val_masks)

drop_last = False

# Create DataLoaders
train_DL = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, drop_last=drop_last, num_workers=2) 
test_DL = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=drop_last, num_workers=2)
val_DL = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=drop_last, num_workers=2)


in_channels = 1
out_channels = 1

model = CompletionNet_CutOnRawE_300().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

# lambda function to change learing rate at epoch = 4 (not used)
# Function to update learning rate based on epoch (not used)
def update_learning_rate(optimizer, initial_lr, actual_epoch, threshold, scale_factor):
    if actual_epoch < threshold:
        new_lr = initial_lr
    else:
        new_lr = initial_lr * scale_factor
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


start_epoch = 0
if args.resume:
    if os.path.isfile(args.checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, path=args.checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"No check point found at {args.checkpoint_path}. Starting new training from scratch.")

use_energy=False # run with enerdy deposition mode? ie also learn to predict energy depositions? 

def train(model, train_loader, optimizer, epoch, use_energy=use_energy):
    model.train()
    total_loss = 0
    loss_cls_total = 0
    loss_energy_total = 0
    count = 0
    chamfer_dist_list = []
    emd_dist_list = []
    DC = []

    for batch_idx, (input_coords, input_feats, target_coords, target_feats, masks, target_l) in enumerate(train_loader):
        s = time() # time is not being used
        optimizer.zero_grad()
        
        # generate input tensor
        input_tensor = ME.SparseTensor(features=input_feats, coordinates=input_coords, device=device)
        
        # generate target tensor
        target_tensor = ME.SparseTensor(features=target_feats, coordinates=target_coords, device=device)
        
        # generate target tensor key for inputs
        cm = input_tensor.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(target_l).to(device), 
            string_id = "target_inputs"
        )
        
        out_cls, targets, output_tensor, loss_cls, loss_energy, losst = model(input_tensor, target_key, target_tensor, val_train=False, val_test=False, test=False, use_energy=use_energy)
        print(len(input_tensor), "->", len(output_tensor), "/", len(target_coords), 'target_coords_list: ', len(target_l))
        track_ids, chamfer_dist, emd_dist = calculate_distances(output_tensor.C.detach().cpu().numpy(), target_tensor.C.detach().cpu().numpy())
        chamfer_dist_list.append(chamfer_dist)
        emd_dist_list.append(emd_dist)
        
        losst.backward()
               
        optimizer.step()
        
        # Update lr
        #update_learning_rate(optimizer, lr, epoch, threshold=4, scale_factor=1e-2)
        for param_group in optimizer.param_groups:
            print(f"Epoch {epoch}, Learning Rate: {param_group['lr']}")
        
        t = time() - s
        #print(f"Iter: {batch_idx}, Loss: {losst.item():.3f}, Data Loading Time: {d:.3f}, Total Time: {t:.3f}")
        print(f"Train: Iter: {batch_idx}, Loss: {losst.item():.3f}")
      
        total_loss += losst.item()
        loss_cls_total += loss_cls.item()
        loss_energy_total += loss_energy.item()
        
        del losst, loss_cls, loss_energy, output_tensor, input_tensor, targets, out_cls, target_key, target_tensor
        gc.collect()
        torch.cuda.empty_cache()
        count += 1

    average_loss = total_loss / count if count > 0 else 0
    average_cls_loss = loss_cls_total / count if count > 0 else 0
    average_energy_loss = loss_energy_total/ count if count > 0 else 0
    chamfer_dist_epoch = np.mean(chamfer_dist_list)
    emd_dist_epoch = np.mean(emd_dist_list)
    
    return average_loss, average_cls_loss, average_energy_loss, chamfer_dist_epoch, emd_dist_epoch

def validate(model, val_loader, val_train=False, val_test=False, optimizer=None):
    model.eval()
    total_loss = 0
    loss_cls_total = 0
    loss_energy_total = 0
    count = 0
    chamfer_dist_list = []
    emd_dist_list = []
    DC = []
    with torch.no_grad():
        for batch_idx, (input_coords, input_feats, target_coords, target_feats, masks, target_l) in enumerate(val_loader):
            s = time()

            # generate input tensor
            input_tensor = ME.SparseTensor(features=input_feats, coordinates=input_coords, device=device)
            
            # generate target tensor
            target_tensor = ME.SparseTensor(features=target_feats, coordinates=target_coords, device=device)

            # generate target tensor
            cm = input_tensor.coordinate_manager
            target_key, _ = cm.insert_and_map(
                ME.utils.batched_coordinates(target_l).to(device), 
                string_id = "target"
            )
            
            # generate target tensor for comparison? 
            #tg_tensor_comp = ME.SparseTensor(features=target_feats, coordinates=target_coords, device=device)
            
            # only using val_train right now; val_test returns zero for loss always, as it should -- ie it is the same as "test"
            if val_train:
                out_cls, targets, output_tensor, loss_cls, loss_energy, losst = model(input_tensor, target_key, target_tensor, val_train=val_train, val_test=val_test, test=False, use_energy=use_energy)
            elif val_test:
                out_cls, targets, output_tensor, loss_cls, loss_energy, losst = model(input_tensor, 0, 0,  val_train=val_train, val_test=val_test, test=False) # this logic of passing in 0 for target_tensor might be flawed, but only val_train is being used anyway
                
            print(len(input_tensor), "->", len(output_tensor), "/", len(target_coords), 'target_coords_list: ', len(target_l))
            track_ids, chamfer_dist, emd_dist = calculate_distances(output_tensor.C.detach().cpu().numpy(), target_tensor.C.detach().cpu().numpy())
            chamfer_dist_list.append(chamfer_dist)
            emd_dist_list.append(emd_dist)

            t = time() - s
            #print(f"Iter: {batch_idx}, Loss: {losst.item():.3f}, Data Loading Time: {d:.3f}, Total Time: {t:.3f}")
            if val_train:
                
                print(f"Validation: Iter: {batch_idx}, Loss: {losst.item():.3f}")

                total_loss += losst.item()
                loss_cls_total += loss_cls.item()
                loss_energy_total += loss_energy.item()
            
            elif val_test: # again this may not make sense, but only val_train is being used
                losst_val_test = torch.tensor(losst)
                losst = losst_val_test

                print(f"Validation: Iter: {batch_idx}, Loss: {losst.item():.3f}")

                total_loss += losst.item()   
                loss_cls_total += loss_cls.item()
                loss_energy_total += loss_energy.item()
                
            del losst, loss_cls, loss_energy, output_tensor, input_tensor, targets, out_cls, target_key, target_tensor
        
            gc.collect()
            torch.cuda.empty_cache()
            count += 1

    average_loss = total_loss / count if count > 0 else 0
    average_cls_loss = loss_cls_total / count if count > 0 else 0
    average_energy_loss = loss_energy_total/ count if count > 0 else 0
    chamfer_dist_epoch = np.mean(chamfer_dist_list)
    emd_dist_epoch = np.mean(emd_dist_list)

    return average_loss, average_cls_loss, average_energy_loss, chamfer_dist_epoch, emd_dist_epoch

def test(model, test_loader, val_train=False, val_test=False, optimizer=None):
    model.eval()
    total_loss = 0
    count = 0
    input_coords_l = []
    predictions_coords = [] # these really just input coords (sanity check)
    input_feats_l = []
    predictions_energy = []
    target_coords_l = [] # wont be used downstream
    target_energy_l = []
    true_energies = []
    chamfer_dist_list = []
    emd_dist_list = []
    DC = []
    with torch.no_grad():
        for batch_idx, (input_coords, input_feats, target_coords, target_feats, masks, target_l) in enumerate(test_loader):
            s = time()

            # generate input tensor
            input_tensor = ME.SparseTensor(features=input_feats, coordinates=input_coords, device=device)
            
            # generate target tensor for comparison? 
            #tg_tensor_comp = ME.SparseTensor(features=target_feats, coordinates=target_coords, device=device)
            
            # generate target tensor
            cm = input_tensor.coordinate_manager
          
            out_cls, targets, output_tensor, loss_cls, loss_energy, losst = model(input_tensor, None, None, val_train=val_train, val_test=val_test, test=True, use_energy=use_energy)
            #output_tensor.F[:] = 1
            #print(targets)
            print(len(input_tensor), "->", len(output_tensor), "/", len(target_coords), 'target_coords_list: ', len(target_l))

            # generate target tensor
            target_tensor = ME.SparseTensor(features=target_feats, coordinates=target_coords, device=device)

            track_ids, chamfer_dist, emd_dist = calculate_distances(output_tensor.C.detach().cpu().numpy(), target_tensor.C.detach().cpu().numpy())
            chamfer_dist_list.append(chamfer_dist)
            emd_dist_list.append(emd_dist)
            
            t = time() - s

            input_coords_l.append(input_tensor.C.detach().cpu().numpy())
            target_coords_l.append(target_coords.detach().cpu().numpy())
            predictions_coords.append(output_tensor.C.detach().cpu().numpy())
            input_feats_l.append(input_tensor.F.detach().cpu().numpy())
          
            target_energy_l.append(target_feats.detach().cpu().numpy())
            predictions_energy.append(output_tensor.F.detach().cpu().numpy())

            del losst, loss_cls, loss_energy, output_tensor, input_tensor, targets, out_cls
            gc.collect()
            torch.cuda.empty_cache()
            count += 1
            
    chamfer_dist_epoch = np.mean(chamfer_dist_list)
    emd_dist_epoch = np.mean(emd_dist_list)
            
    return predictions_coords, predictions_energy, input_coords_l, input_feats_l, target_coords_l, target_energy_l, chamfer_dist_epoch, emd_dist_epoch


# Early stopping parameters
patience = 10*5  # Number of epochs to wait for improvement; does not matter at the moment 
best_val_loss = float('inf')
epochs_no_improve = 0
warm_up_epochs = 50
            
for epoch in range(start_epoch, num_epochs):
    train_loss, train_cls_loss, train_energy_loss, train_chamfer_dist, train_emd_dist = train(model, train_DL, optimizer, epoch)
    val_loss, val_cls_loss, val_energy_loss, val_chamfer_dist, val_emd_dist = validate(model, val_DL, val_train=True, val_test=False)
    predictions_coords, predictions_energy, input_coords, input_feats, target_coords, target_energy, test_chamfer_dist, test_emd_dist = test(model, test_DL, val_train=False, val_test=False)
    base_dir = '/path/to/base_dir/'
    save_predictions(base_dir, predictions_coords, predictions_energy, input_coords, input_feats, target_coords, target_energy, epoch)
    
    # Log losses to W&B
    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_chamfer": train_chamfer_dist, "train_emd": train_emd_dist, "train_energy_loss": train_energy_loss, "val_loss": val_loss, "val_chamfer": val_chamfer_dist, "val_emd": val_emd_dist, "val_energy_loss": val_energy_loss, "test_chamfer": test_chamfer_dist, "test_emd": test_emd_dist})

    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save checkpoint for the best model
        save_checkpoint(model, optimizer, epoch, path=f'/path/to/best_checkpoint.pth')
        
    else:
        epochs_no_improve += 1
    
    # Save checkpoint every 1 epochs
    if epoch % 1 == 0 or epoch == num_epochs - 1:
        save_checkpoint(model, optimizer, epoch, path=f'/path/to/checkpoint_epoch_{epoch}.pth')
        
    #if epoch % print_every == 0:
    print(f'Epoch {epoch} Training Loss: {train_loss:.4f}')
    print(f'Epoch {epoch} Validation Loss: {val_loss:.4f}')
    print(f'Epoch {epoch} Training chamfer, emd: {train_chamfer_dist:.4f}, {train_emd_dist:.4f}')
    print(f'Epoch {epoch} Validation chamfer, emd: {val_chamfer_dist:.4f}, {val_emd_dist:.4f}')
    print(f'Epoch {epoch} Testing chamfer, emd: {test_chamfer_dist:.4f}, {test_emd_dist:.4f}')
    
# End run on W&B
wandb.finish()

       
