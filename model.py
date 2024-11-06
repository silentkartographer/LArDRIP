# Part of this code is taken from: https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/completion.py

# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

# Please cite H. Utaegbulam, “LArDRIP,” https://github.com/
# silentkartographer/LArDRIP if you use any part of the code. 

import gc, argparse
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import MinkowskiEngine as ME
from time import time
import torch.nn.init as init
import ot
from scipy.spatial.distance import cdist
device = ('cuda' if torch.cuda.is_available() else 'cpu')


class CompletionNet_CutOnRawE_300_Energy(nn.Module):

    ENC_CHANNELS = [22, 58, 84, 158, 312, 676, 1024, 1576, 2048]
    #ENC_CHANNELS = [22, 150, 300, 578, 1024, 1778, 2048]
    DEC_CHANNELS = ENC_CHANNELS

    def __init__(self):
        nn.Module.__init__(self)
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS

        # Encoder
        self.enc_block_s1 = nn.Sequential(
            ME.MinkowskiConvolution(1, enc_ch[0], kernel_size=3, stride=1, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )
        #
        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s64s128 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[7], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[7]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[7], enc_ch[7], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[7]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.enc_block_s128s256 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[7], enc_ch[8], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[8]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[8], enc_ch[8], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[8]),
            ME.MinkowskiReLU(),
            # ME.MinkowskiDropout(0.3),
        )

        # Decoder
        self.dec_block_s256s128 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[8],
                dec_ch[7],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[7]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[7], dec_ch[7], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[7]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s128_cls = ME.MinkowskiConvolution(dec_ch[7], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s128s64 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[7],
                dec_ch[6],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[6]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[6], dec_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[6]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s64_cls = ME.MinkowskiConvolution(dec_ch[6], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s32s16 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[5],
                dec_ch[4],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[4], dec_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[4]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s16s8 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[4],
                dec_ch[3],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[3], dec_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[3]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(dec_ch[3], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s8s4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[3],
                dec_ch[2],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[2], dec_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s4s2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[2],
                dec_ch[1],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[1], dec_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=3)

        self.dec_block_s2s1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                dec_ch[1],
                dec_ch[0],
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[0], dec_ch[0], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU(),
            #ME.MinkowskiDropout(0.3),
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3)

        self.final_out_cls = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiSigmoid(),
        )

        self.final_out_energy = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3),
            #ME.MinkowskiSigmoid(),
        )

        # pruning
        self.pruning = ME.MinkowskiPruning()
        
        # losses and weights
        self.crit_cls = nn.BCEWithLogitsLoss()
        self.crit_energy = nn.MSELoss()
        self.weight_cls_focus_on_cls = 0.7
        self.weight_energy_focus_on_cls = 0.3
        self.softplus = nn.Softplus(beta=1, threshold=3)


    def earth_movers_distance(self, track1, track2):
        # this currently does not respect autograd/ backpropogation, maybe try geomloss
        cost_matrix = cdist(track1, track2)
        a = np.ones((track1.shape[0],)) / track1.shape[0]
        b = np.ones((track2.shape[0],)) / track2.shape[0]
        emd = ot.emd2(a, b, cost_matrix)
        return emd

    def chamfer_dist(self, track1_full, track2_full, xmin=114, xmax=128, ymin=140, ymax=154):
        # track1_full is prediction! track2_full is target!

        # track1 and track2 should be 2D torch tensors of shape (B, N, D) where N is the number of voxels and D is dimension (3 for XYZ); B is batch or track id
        #print(torch.unique(track1_full[:, 0]).size(0), torch.unique(track2_full[:, 0]).size(0))
        #assert torch.unique(track1_full[:, 0]).all() == torch.unique(track2_full[:, 0]).all(), "Prediction and Target MUST have the same number of tracks"
        #assert torch.equal(torch.unique(track1_full[:, 0]), torch.unique(
        #    track2_full[:, 0])), "Prediction and Target MUST have the same number of tracks and track IDs"

        if not torch.equal(torch.unique(track1_full[:, 0]), torch.unique(track2_full[:, 0])):
            print(f"Predictions and Targets do not have the same amount of tracks. Hopefully this is in a very early epoch ..."
                  , f"{torch.unique(track1_full[:, 0]).size(0), torch.unique(track2_full[:, 0]).size(0)}")

        chamfer_dist_vals = []
        track_ids = torch.unique(track1_full[:,0])
        for track_id in track_ids:

            track_mask1 = (track1_full[:, 0] == track_id)
            track_mask2 = (track2_full[:, 0] == track_id)



            mask_inactive_region_track1 = (
                    ((track1_full[:, 1] >= xmin) & (track1_full[:, 1] <= xmax)) |
                    ((track1_full[:, 2] >= ymin) & (track1_full[:, 2] <= ymax))
            )
            mask_inactive_region_track2 = (
                    ((track2_full[:, 1] >= xmin) & (track2_full[:, 1] <= xmax)) |
                    ((track2_full[:, 2] >= ymin) & (track2_full[:, 2] <= ymax))
            )

            track1 = track1_full[mask_inactive_region_track1 & track_mask1][:, 1:]  # Remove track ID, keep only XYZ
            track2 = track2_full[mask_inactive_region_track2 & track_mask2][:, 1:]  # Remove track ID, keep only XYZ

            # If either track is empty, use the full tracks
            if track1.size(0) == 0 or track2.size(0) == 0:
                track1 = track1_full[track_mask1][:, 1:]  # Remove track ID, keep only XYZ
                track2 = track2_full[track_mask2][:, 1:]  # Remove track ID, keep only XYZ

            dist1 = torch.mean(torch.min(torch.cdist(track1, track2), dim=1)[0])
            dist2 = torch.mean(torch.min(torch.cdist(track2, track1), dim=1)[0])
            chamfer_dist_vals.append(dist1 + dist2)

        chamfer_dist_vals_tensor = torch.tensor(chamfer_dist_vals)

        return torch.mean(chamfer_dist_vals_tensor)


    def sinkhorn_loss(self, track1_full, track2_full, xmin=114, xmax=128, ymin=140, ymax=154, epsilon=0.001, max_iters=350, tol=1e-8):
        """
        Computes the Sinkhorn loss (approximate Earth Mover's Distance) between two point clouds (tracks).

        Args:
            track1_full (torch.Tensor): Tensor of shape (N, D) for the first track.
            track2_full (torch.Tensor): Tensor of shape (M, D) for the second track.
            xmin, xmax, ymin, ymax: Bounds for filtering tracks in the inactive region.
            epsilon (float): Regularization term for the Sinkhorn distance.
            max_iters (int): Maximum number of iterations for the Sinkhorn algorithm.
            tol (float): Convergence tolerance for the Sinkhorn iterations.

        Returns:
            torch.Tensor: The average Sinkhorn loss between all tracks.
        """
        # Ensure track IDs are aligned between predictions and targets
        #assert torch.unique(track1_full[:, 0]).all() == torch.unique(track2_full[:, 0]).all(), "Prediction and Target MUST have the same number of tracks"

        if not torch.equal(torch.unique(track1_full[:, 0]), torch.unique(track2_full[:, 0])):
            print(f"Predictions and Targets do not have the same amount of tracks. Hopefully this is in a very early epoch ..."
                  , f"{torch.unique(track1_full[:, 0]).size(0), torch.unique(track2_full[:, 0]).size(0)}")

        sinkhorn_loss_vals = []
        track_ids = torch.unique(track1_full[:, 0])

        for track_id in track_ids:
            track_mask1 = (track1_full[:, 0] == track_id)
            track_mask2 = (track2_full[:, 0] == track_id)

            mask_inactive_region_track1 = (
                    ((track1_full[:, 1] >= xmin) & (track1_full[:, 1] <= xmax)) |
                    ((track1_full[:, 2] >= ymin) & (track1_full[:, 2] <= ymax))
            )
            mask_inactive_region_track2 = (
                    ((track2_full[:, 1] >= xmin) & (track2_full[:, 1] <= xmax)) |
                    ((track2_full[:, 2] >= ymin) & (track2_full[:, 2] <= ymax))
            )

            track1 = track1_full[mask_inactive_region_track1 & track_mask1]
            track2 = track2_full[mask_inactive_region_track2 & track_mask2]

            # If either track is empty, use the full tracks
            if track1.size(0) == 0 or track2.size(0) == 0:
                track1 = track1_full[track_mask1]
                track2 = track2_full[track_mask2]

            # Calculate pairwise cost matrix (distance matrix)
            C = torch.cdist(track1[:, 1:], track2[:, 1:], p=2)  # Exclude track ID for distance computation

            # Normalize the distance matrix
            #C /= C.max()
            C = C / (C.max() + 1e-9) # Adding a small epsilon to prevent division by zero

            # Initial uniform weights
            mu = torch.ones(track1.size(0), device=track1.device) / track1.size(0)
            nu = torch.ones(track2.size(0), device=track2.device) / track2.size(0)

            # Initialize the dual variables
            u = torch.zeros_like(mu)
            v = torch.zeros_like(nu)

            # Sinkhorn iterations
            for _ in range(max_iters):
                u_prev = u
                v_prev = v
                u = epsilon * (torch.log(mu) - torch.logsumexp(-C / epsilon + v.unsqueeze(0), dim=1)) + u
                v = epsilon * (torch.log(nu) - torch.logsumexp(-C / epsilon + u.unsqueeze(1), dim=0)) + v

                # check convergence
                if torch.max(torch.abs(u - u_prev)) < tol and torch.max(torch.abs(v - v_prev)) < tol:
                    break

            # Compute the approximate Sinkhorn distance
            pi = torch.exp(-C / epsilon + u.unsqueeze(1) + v.unsqueeze(0))
            sinkhorn_dist = torch.sum(pi * C)

            sinkhorn_loss_vals.append(sinkhorn_dist)

        #sinkhorn_loss_vals_tensor = torch.tensor(sinkhorn_loss_vals, device=track1_full.device)

        #return torch.mean(sinkhorn_loss_vals_tensor)
        return torch.mean(torch.stack(sinkhorn_loss_vals))




    def get_target(self, out, target_key, kernel_size=1): # target_key is actually target_key_inputs
        with torch.no_grad():
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            target = torch.zeros(len(out), dtype=torch.bool, device=device)
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target
        
    def masked_loss_energy_real_chunk(self, dec_sx_cls, target, target_coords, crit_energy, device, xmin=114, xmax=128, ymin=140, ymax=154, use_de_dx_loss=False, calculate_in_inactive=True):
        # Convert target to match the data type of the predictions and move to the correct device
        target = target.type(dec_sx_cls.F.dtype).to(device)

        # Mask for predictions with non-zero features
        mask_non_zero_prediction = (dec_sx_cls.F.squeeze() != 0)

        if calculate_in_inactive:
            # Create a mask for target and dec_sx_cls coords within the specified limits
            mask_inactive_region_prediction = (
                ((dec_sx_cls.C[:, 1] >= xmin) & (dec_sx_cls.C[:, 1] <= xmax)) |
                ((dec_sx_cls.C[:, 2] >= ymin) & (dec_sx_cls.C[:, 2] <= ymax))
            )
            mask_inactive_region_target = (
                ((target_coords[:, 1] >= xmin) & (target_coords[:, 1] <= xmax)) |
                ((target_coords[:, 2] >= ymin) & (target_coords[:, 2] <= ymax))
            )
            mask_prediction_nonzero_and_inactive = mask_non_zero_prediction & mask_inactive_region_prediction
        else:
            mask_prediction_nonzero_and_inactive = mask_non_zero_prediction  # Use non-zero prediction mask only

        # Apply the mask to predictions and targets
        pred_coords_masked = dec_sx_cls.C[mask_prediction_nonzero_and_inactive]
        pred_features_masked = dec_sx_cls.F[mask_prediction_nonzero_and_inactive]
        target_coords_masked = target_coords[mask_inactive_region_target]
        target_features_masked = target[mask_inactive_region_target]

        if pred_coords_masked.size(0) == 0 or target_coords_masked.size(0) == 0:
            # If there are no valid coordinates in the region, use the full track for both predictions and targets
            
            print("Using full region")
            
            pred_coords_masked = dec_sx_cls.C
            pred_features_masked = dec_sx_cls.F
            target_coords_masked = target_coords
            target_features_masked = target

            # Ensure we only compare coordinates from the same batch/track
            batch_numbers = pred_coords_masked[:, 0].unique()
        else:
            print("Using inactive region")
            # Ensure we only compare coordinates from the same batch/track
            batch_numbers = pred_coords_masked[:, 0].unique()
        
        total_loss_energy = 0.0
        count = 0

        # Iterate through each batch/track
        for batch in batch_numbers:
            # Filter predicted and target coordinates for the current batch
            pred_batch_mask = pred_coords_masked[:, 0] == batch
            target_batch_mask = target_coords_masked[:, 0] == batch

            pred_coords_batch = pred_coords_masked[pred_batch_mask][:, 1:]  # Ignore the batch column
            pred_features_batch = pred_features_masked[pred_batch_mask]
            target_coords_batch = target_coords_masked[target_batch_mask][:, 1:]  # Ignore the batch column
            target_features_batch = target_features_masked[target_batch_mask]

            if pred_coords_batch.size(0) == 0 or target_coords_batch.size(0) == 0:
                continue

            # Expand dimensions to calculate distances between all pairs
            pred_coords_expand = pred_coords_batch.float().unsqueeze(1).expand(-1, target_coords_batch.size(0), -1)
            target_coords_expand = target_coords_batch.float().unsqueeze(0).expand(pred_coords_batch.size(0), -1, -1)

            # Calculate squared distances between each pair of voxels
            distances = torch.norm(pred_coords_expand - target_coords_expand, dim=2)

            # Find the index of the closest target voxel for each predicted voxel
            closest_target_indices = torch.argmin(distances, dim=1)

            # Gather the features of the closest target voxels
            closest_target_features = target_features_batch[closest_target_indices]

            # Calculate the MSE loss between predicted features and closest target features
            loss = crit_energy(pred_features_batch, closest_target_features)
            total_loss_energy += loss.item()
            count += 1
        
        if count > 0:
            loss_energy = torch.tensor(total_loss_energy / count, device=device)
        else:
            loss_energy = torch.tensor(0.0, device=device)  

        ### dE/dx loss
        energy_grad_loss = 0.0
        chunk_size = 1  # Adjust chunk size based on available memory

        # Apply the mask to predictions and targets
        pred_coords_masked = dec_sx_cls.C[mask_prediction_nonzero_and_inactive]
        pred_features_masked = dec_sx_cls.F[mask_prediction_nonzero_and_inactive]

        if use_de_dx_loss and pred_coords_masked.size(0) != 0:
            # Get the batch numbers (track numbers) for masked voxels
            batch_numbers = pred_coords_masked[:, 0].unique()  # Unique track numbers

            for batch in batch_numbers:
                # Filter masked coordinates and predictions for the current batch
                batch_mask = pred_coords_masked[:, 0] == batch
                masked_coords_batch = pred_coords_masked[batch_mask][:, 1:].float()  # Ignore the batch column
                masked_predictions_batch = pred_features_masked[batch_mask]

                if masked_coords_batch.size(0) == 0:
                    continue

                # Initialize an empty tensor to store the distances between voxels
                distances = torch.empty(masked_coords_batch.size(0), masked_coords_batch.size(0), device=device)

                # Chunked calculation of pairwise distances
                for i in range(0, masked_coords_batch.size(0), chunk_size):
                    for j in range(0, masked_coords_batch.size(0), chunk_size):
                        # Calculate the pairwise distances for this chunk
                        distances_chunk = torch.norm(
                            masked_coords_batch[i:i + chunk_size].unsqueeze(1) -
                            masked_coords_batch[j:j + chunk_size].unsqueeze(0), dim=2
                        )
                        # Assign the calculated chunk back to the appropriate section in the distances tensor
                        distances[i:i + chunk_size, j:j + chunk_size] = distances_chunk

                # Find the index of the closest neighbor voxel for each voxel
                closest_voxel_indices = torch.argmin(distances + torch.eye(distances.size(0), device=device) * 1e6, dim=1)

                # Gather the predictions of the closest voxels
                closest_voxel_predictions = masked_predictions_batch[closest_voxel_indices]

                # Calculate the change in energy (dE) for each pair of adjacent voxels
                energy_changes = torch.abs(masked_predictions_batch - closest_voxel_predictions)

                # Compute the distance between each voxel and its closest neighbor
                closest_distances = distances[torch.arange(distances.size(0)), closest_voxel_indices]

                # Compute the expected change in energy per distance (dE/dx)
                dx = closest_distances * 0.38  # Actual distance between voxels (3.8 mm or 0.38 cm per unit step)
                predicted_de_dx = energy_changes / dx

                # Known relationship: abs(dE/dx) <= 2.1 MeV/cm
                # Penalize the predicted dE/dx if it exceeds the limit of 2.1 MeV/cm
                de_dx_penalty = torch.clamp(predicted_de_dx - 2.1, min=0)  # Only penalize if dE/dx exceeds 2.1

                # Chunk the penalty calculation to avoid OOM
                for i in range(0, de_dx_penalty.size(0), chunk_size):
                    de_dx_penalty_chunk = de_dx_penalty[i:i + chunk_size]
                    energy_grad_loss += crit_energy(de_dx_penalty_chunk, torch.zeros_like(de_dx_penalty_chunk))

            # Normalize the energy gradient loss by the number of tracks (batch numbers)
            if batch_numbers.numel() != 0:
                energy_grad_loss /= batch_numbers.numel()
            else:
                energy_grad_loss = torch.tensor(0.0, device=device)

        # Combine the classification loss and the energy gradient loss
        total_loss = loss_energy + energy_grad_loss if use_de_dx_loss else loss_energy

        return total_loss

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def get_keep_vec(self, cls, res):
        """
        To ensure that a sparse tensor can safely be converted to the dense one.
        """
        a = (cls.F > 0).squeeze()
        b = (cls.C[:, 1] < res[2]).squeeze()
        c = (cls.C[:, 2] < res[3]).squeeze()
        d = (cls.C[:, 3] < res[4]).squeeze()
        ab = torch.logical_and(a, b)
        abc = torch.logical_and(ab, c)
        abcd = torch.logical_and(abc, d)
        return abcd

    def forward(self, partial_in, target_key, target_tensor, val_train=False, val_test=False, test=False, use_energy=False, use_chamfer=True, use_emd=True):
        #loss_cls = 0
        #loss_energy = 0
        total_loss = 0 
        loss_energy = torch.tensor(0.0)
        loss_cls = torch.tensor(0.0)
        out_cls, targets = [], []
        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)
        enc_s128 = self.enc_block_s64s128(enc_s64)
        enc_s256 = self.enc_block_s128s256(enc_s128)


        # ##################################################
        # # Decoder 256 -> 128
        # ##################################################
        dec_s128 = self.dec_block_s256s128(enc_s256) + enc_s128
        # Add encoder features
        dec_s128_cls = self.dec_s128_cls(dec_s128)
        keep_s128 = (dec_s128_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s128,
                                     target_key)  # checks if target voxels exists where voxels are generated in dec_sx; returns 1 if so and 0 if not. returns a new tensor (not sparse tensor)
            # target_tensor_for_energy_loss_energies, target_tensor_for_energy_loss_coords = self.get_target_values(target_tensor, target_key_targets) # returns a new tensor (not sparse tensor) filled with target values (E) at corresponding target voxels
            loss_cls = self.crit_cls(dec_s128_cls.F.squeeze(), target.type(dec_s128_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s128_cls, target_tensor.F, target_tensor.C,
            #                                                     self.crit_energy, device)
            # weighting then adding
            #total_loss += self.weight_cls_focus_on_cls * loss_cls + self.weight_energy_focus_on_cls * loss_energy if use_energy else loss_cls

            total_loss += loss_cls

            # if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s32_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s128 += target
        elif val_train:
            keep_s128 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s32

        dec_s64 = self.pruning(dec_s128, keep_s128)
        del keep_s128, target, dec_s128_cls
        gc.collect()
        torch.cuda.empty_cache()

        print("finished first part 300")

        # ##################################################
        # # Decoder 128 -> 64
        # ##################################################
        dec_s64 = self.dec_block_s128s64(enc_s128) + enc_s64
        # Add encoder features
        dec_s64_cls = self.dec_s64_cls(dec_s64)
        keep_s64 = (dec_s64_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s64,
                                     target_key)  # checks if target voxels exists where voxels are generated in dec_sx; returns 1 if so and 0 if not. returns a new tensor (not sparse tensor)
            # target_tensor_for_energy_loss_energies, target_tensor_for_energy_loss_coords = self.get_target_values(target_tensor, target_key_targets) # returns a new tensor (not sparse tensor) filled with target values (E) at corresponding target voxels
            loss_cls = self.crit_cls(dec_s64_cls.F.squeeze(), target.type(dec_s64_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s64_cls, target_tensor.F, target_tensor.C,
            #                                                     self.crit_energy, device)
            # weighting then adding
            #total_loss += self.weight_cls_focus_on_cls * loss_cls + self.weight_energy_focus_on_cls * loss_energy if use_energy else loss_cls
            total_loss += loss_cls

            # if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s32_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s64 += target
        elif val_train:
            keep_s64 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s32

        dec_s64 = self.pruning(dec_s64, keep_s64)
        del keep_s64, target, dec_s64_cls
        gc.collect()
        torch.cuda.empty_cache()

        #print("finished first part")



        # ##################################################
        # # Decoder 64 -> 32
        # ##################################################
        dec_s32 = self.dec_block_s64s32(dec_s64) + enc_s32
        # Add encoder features
        dec_s32_cls = self.dec_s32_cls(dec_s32)
        #print(dec_s32_cls.F)
        keep_s32 = (dec_s32_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s32, target_key) # checks if target voxels exists where voxels are generated in dec_sx; returns 1 if so and 0 if not. returns a new tensor (not sparse tensor)
            #target_tensor_for_energy_loss_energies, target_tensor_for_energy_loss_coords = self.get_target_values(target_tensor, target_key_targets) # returns a new tensor (not sparse tensor) filled with target values (E) at corresponding target voxels
            loss_cls = self.crit_cls(dec_s32_cls.F.squeeze(), target.type(dec_s32_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s32_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # weighting then adding 
            #total_loss += self.weight_cls_focus_on_cls * loss_cls + self.weight_energy_focus_on_cls * loss_energy if use_energy else loss_cls

            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s32_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s32 += target
        elif val_train:
            keep_s32 += target
        elif val_test:
            pass
        elif test:
            pass

            
        # Remove voxels s32

        dec_s32 = self.pruning(dec_s32, keep_s32)
        del keep_s32, target, dec_s32_cls
        gc.collect()
        torch.cuda.empty_cache()
        
        #print("finished first part")
        # ##################################################
        # # Decoder 32 -> 16
        # ##################################################
        dec_s16 = self.dec_block_s32s16(dec_s32) + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)
        keep_s16 = (dec_s16_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s16, target_key)
            loss_cls = self.crit_cls(dec_s16_cls.F.squeeze(), target.type(dec_s16_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s16_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # weighting then adding 
            #total_loss += self.weight_cls_focus_on_cls * loss_cls + self.weight_energy_focus_on_cls * loss_energy if use_energy else loss_cls
            
            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s16_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s16 += target
        elif val_train:
            keep_s16 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s16
        dec_s16 = self.pruning(dec_s16, keep_s16)
        del keep_s16, target, dec_s16_cls
        gc.collect()
        torch.cuda.empty_cache()

        # ##################################################
        # # Decoder 16 -> 8
        # ##################################################
        dec_s8 = self.dec_block_s16s8(dec_s16) + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)
        keep_s8 = (dec_s8_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s8, target_key)
            loss_cls = self.crit_cls(dec_s8_cls.F.squeeze(), target.type(dec_s8_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s8_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # weighting then adding 
            #total_loss += self.weight_cls_focus_on_cls * loss_cls + self.weight_energy_focus_on_cls * loss_energy if use_energy else loss_cls
            
            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s8_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s8 += target
        elif val_train:
            keep_s8 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s8
        dec_s8 = self.pruning(dec_s8, keep_s8)
        del keep_s8, target, dec_s8_cls
        gc.collect()
        torch.cuda.empty_cache()

        # ##################################################
        # # Decoder 8 -> 4
        # ##################################################
        dec_s4 = self.dec_block_s8s4(dec_s8) + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)
        keep_s4 = (dec_s4_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s4, target_key)
            loss_cls = self.crit_cls(dec_s4_cls.F.squeeze(), target.type(dec_s4_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s4_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # NOT weighting then adding 
            #total_loss += loss_cls + loss_energy if use_energy else loss_cls

            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s4_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s4 += target
        elif val_train:
            keep_s4 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s4
        dec_s4 = self.pruning(dec_s4, keep_s4)
        del keep_s4, target, dec_s4_cls
        gc.collect()
        torch.cuda.empty_cache()

        # ##################################################
        # # Decoder 4 -> 2
        # ##################################################
        dec_s2 = self.dec_block_s4s2(dec_s4) + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)
        keep_s2 = (dec_s2_cls.F > 0).squeeze()

        target = 0
        if target_key is not None:
            target = self.get_target(dec_s2, target_key)
            loss_cls = self.crit_cls(dec_s2_cls.F.squeeze(), target.type(dec_s2_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s2_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # weighting then adding 
            #total_loss += loss_cls + loss_energy if use_energy else loss_cls
            
            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s2_cls.C.float(), target_tensor.C.float())

        if self.training:
            keep_s2 += target
        elif val_train:
            keep_s2 += target
        elif val_test:
            pass
        elif test:
            pass

        # Remove voxels s2
        dec_s2 = self.pruning(dec_s2, keep_s2)
        del keep_s2, target, dec_s2_cls
        gc.collect()
        torch.cuda.empty_cache()

        ##################################################
        # Decoder 2 -> 1
        ##################################################
        dec_s1 = self.dec_block_s2s1(dec_s2) + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        if target_key is not None:
            loss_cls = self.crit_cls(dec_s1_cls.F.squeeze(), self.get_target(dec_s1, target_key).type(dec_s1_cls.F.dtype).to(device))
            #if use_energy:
            #    loss_energy = self.masked_loss_energy_real_chunk(dec_s1_cls, target_tensor.F, target_tensor.C, self.crit_energy, device)
            # weighting then adding 
            #total_loss += loss_cls + loss_energy if use_energy else loss_cls
            
            total_loss += loss_cls
            #if use_chamfer:
            #    total_loss += self.chamfer_dist(dec_s1_cls.C.float(), target_tensor.C.float())

        # Last layer does not require adding the target
        # if self.training:
        #     keep_s1 += target
        # Remove voxels s1
        #print(dec_s1_cls.F)
        dec_s1 = self.pruning(dec_s1, (dec_s1_cls.F > 0).squeeze())
        del dec_s1_cls

        torch.cuda.empty_cache()

        if use_energy:
            dec_s1 = self.final_out_energy(dec_s1)
            final_coords = dec_s1.C
            energy_soft_plus = self.softplus(dec_s1.F) # exponential could work well here too
            dec_s1 = ME.SparseTensor(coordinates=final_coords, features=energy_soft_plus, coordinate_manager=dec_s1.coordinate_manager, device=device)
            if target_tensor is not None:
                loss_energy = self.masked_loss_energy_real_chunk(dec_s1, target_tensor.F, target_tensor.C, self.crit_energy, device)
                total_loss += loss_energy
        else:
            dec_s1 = self.final_out_cls(dec_s1)

        if use_chamfer and target_key is not None:
            total_loss += self.chamfer_dist(dec_s1.C.float(), target_tensor.C.float())

        if use_emd and target_key is not None:
            total_loss += self.sinkhorn_loss(dec_s1.C.float(), target_tensor.C.float())

        if target_tensor is None:
            loss_energy = torch.tensor(0.0) # place holder
            loss_cls = torch.tensor(0.0) # place holder
        
        return out_cls, targets, dec_s1, loss_cls, loss_energy, total_loss
