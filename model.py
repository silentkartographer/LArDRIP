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
device = ('cuda' if torch.cuda.is_available() else 'cpu')

class CompletionNet_energy2_5(nn.Module):
    #ENC_CHANNELS = [22, 58, 84, 158, 312, 676, 1024]
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
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
        )

        self.enc_block_s1s2 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[0], enc_ch[1], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[1], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s2s4 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[1], enc_ch[2], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[2], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s4s8 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[2], enc_ch[3], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[3], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s8s16 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[3], enc_ch[4], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[4], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[4]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s16s32 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[4], enc_ch[5], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[5]),
            ME.MinkowskiReLU(),
        )

        self.enc_block_s32s64 = nn.Sequential(
            ME.MinkowskiConvolution(enc_ch[5], enc_ch[6], kernel_size=2, stride=2, dimension=3,),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(enc_ch[6], enc_ch[6], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(enc_ch[6]),
            ME.MinkowskiReLU(),
        )

        # Decoder
        self.dec_block_s64s32 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                enc_ch[6],
                dec_ch[5],
                kernel_size=4,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(dec_ch[5], dec_ch[5], kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(dec_ch[5]),
            ME.MinkowskiReLU(),
        )

        self.dec_s32_cls = ME.MinkowskiConvolution(dec_ch[5], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s32_energy = ME.MinkowskiConvolution(dec_ch[5], 1, kernel_size=1, bias=True, dimension=3)

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
        )

        self.dec_s16_cls = ME.MinkowskiConvolution(dec_ch[4], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s16_energy = ME.MinkowskiConvolution(dec_ch[4], 1, kernel_size=1, bias=True, dimension=3)

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
        )

        self.dec_s8_cls = ME.MinkowskiConvolution(dec_ch[3], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s8_energy = ME.MinkowskiConvolution(dec_ch[3], 1, kernel_size=1, bias=True, dimension=3)

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
        )

        self.dec_s4_cls = ME.MinkowskiConvolution(dec_ch[2], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s4_energy = ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, bias=True, dimension=3)

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
        )

        self.dec_s2_cls = ME.MinkowskiConvolution(dec_ch[1], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s2_energy = ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=3)

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
        )

        self.dec_s1_cls = ME.MinkowskiConvolution(dec_ch[0], 2, kernel_size=1, bias=True, dimension=3)
        self.dec_s1_energy = ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3)

        self.final_out_cls = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 2, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiSigmoid(),
        )

        self.final_out_energy = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiReLU(),
        )
        
        self.pruning = ME.MinkowskiPruning()
        
        ### Energy

        # Define the layers needed for the encoder and decoder
        self.conv_block1 = self.conv_block(1, 16, kernel_size=1, stride=1, dimension=3)
        #self.conv_block1.5 = self.conv_block(8, 16, kernel_size=1, stride=1, dimension=3)
        self.conv_block2 = self.conv_block(16, 16, kernel_size=1, stride=1, dimension=3, use_bias=False) # conn. with deconv block 4 (16 conn w/ 16) = 32
        self.conv_block3 = self.conv_block(16, 32, kernel_size=3, stride=2, dimension=3, use_bias=False)
        self.conv_block4 = self.conv_block(32, 32, kernel_size=1, stride=1, dimension=3, use_bias=False) # conn. with deconv block 3 (32 conn w/ 32) = 64
        self.conv_block5 = self.conv_block(32, 64, kernel_size=3, stride=2, dimension=3, use_bias=False)
        self.conv_block6 = self.conv_block(64, 64, kernel_size=1, stride=1, dimension=3, use_bias=False) # conn. with deconv block 2 (64 conn w/ 64) = 128
        self.conv_block7 = self.conv_block(64, 128, kernel_size=3, stride=2, dimension=3, use_bias=False)
        self.conv_block8 = self.conv_block(128, 128, kernel_size=1, stride=1, dimension=3, use_bias=False) # conn. with deconv block 1 (128 conn w/ 128) = 256
        self.conv_block9 = self.conv_block(128, 256, kernel_size=3, stride=2, dimension=3, use_bias=False)
        self.conv_block10 = self.conv_block(256, 256, kernel_size=1, stride=1, dimension=3, use_bias=False)

        # Attention mechanism and dropout
        self.attention = MultiHeadSelfAttention3D_multihead(256, 128*4, 4*4)  # feat_size, attention_dim, num_heads
        self.dropout_attention = ME.MinkowskiDropout(0.3)
        
        '''
        # Define the layers needed for the decoder
        self.deconv_block1 = self.deconv_block(2048 * 4*4, 1024, kernel_size=3, stride=2, dimension=3) # in_feat, out, ... (in_feat = feat_size * num_heads)
        self.deconv_block2 = self.deconv_block(1024, 512, kernel_size=3, stride=2, dimension=3)
        self.deconv_block3 = self.deconv_block(512, 256, kernel_size=3, stride=2, dimension=3)
        self.deconv_block4 = self.deconv_block(256, 128, kernel_size=3, stride=2, dimension=3)
        self.deconv_block5 = self.deconv_block(128, 64, kernel_size=3, stride=2, dimension=3)
        self.deconv_block6 = self.deconv_block(64, 32, kernel_size=3, stride=2, dimension=3)
        self.deconv_block7 = self.deconv_block(32, 16, kernel_size=3, stride=2, dimension=3)
        self.final_layer = ME.MinkowskiConvolutionTranspose(16, 1, kernel_size=1, stride=1, dimension=3)
        '''
        
        
        
        # Define the layers needed for the decoder
        #self.deconv_block1 = self.deconv_block(256 * 4*4, 128, kernel_size=3, stride=2, dimension=3)  # in_feat, out, ... (in_feat = feat_size * num_heads) # this is for attention
        self.deconv_block1 = self.deconv_block(256, 128, kernel_size=3, stride=2, dimension=3)  # in_feat, out, ... (in_feat = feat_size * num_heads)  
        self.deconv_block2 = self.deconv_block(256, 64, kernel_size=3, stride=2, dimension=3)
        self.deconv_block3 = self.deconv_block(128, 32, kernel_size=3, stride=2, dimension=3)
        self.deconv_block4 = self.deconv_block(64, 16, kernel_size=3, stride=2, dimension=3)
        self.deconv_block5 = self.deconv_block(32, 8, kernel_size=3, stride=2, dimension=3) # not used
        self.final_layer = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(32, 16, kernel_size=1, stride=1, dimension=3),
            ME.MinkowskiConvolutionTranspose(16, 1, kernel_size=1, stride=1, dimension=3)
        )

            
        
    
        
        
        #self.sigmoid = nn.Sigmoid()
        
        # Threshold for RMSE
        self.rmse_threshold = 100000000000000 #0.15
        self.mse_threshold = self.rmse_threshold**2
        
        # Initialize different loss functions, activations
        self.softplus = nn.Softplus(beta=1, threshold=3)
        self.crit_cls = nn.BCEWithLogitsLoss()
        self.crit_energy = nn.MSELoss(reduction='none')
        self.crit_energy_final = nn.MSELoss()
        

    def conv_block(self, in_ch, out_ch, kernel_size, stride, dimension, dropout_rate=0.3, use_bias=True):
        #print(f"Creating MinkowskiConvolution with dimension: {dimension}")

        return nn.Sequential(
            ME.MinkowskiConvolution(in_ch, out_ch, kernel_size, stride, dimension=3, bias=use_bias),
            ME.MinkowskiBatchNorm(out_ch),
            #nn.Softplus(beta=1, threshold=3),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(dropout_rate)
        )

    def deconv_block(self, in_ch, out_ch, kernel_size, stride, dimension, use_bias=True):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_ch, out_ch, kernel_size, stride, dimension=3, bias=use_bias),
            ME.MinkowskiBatchNorm(out_ch),
            #nn.Softplus(beta=1, threshold=3)
            ME.MinkowskiReLU(inplace=True)
        )

    def encoder_stage(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)
        out6 = self.conv_block6(out5)
        out7 = self.conv_block7(out6)
        out8 = self.conv_block8(out7)
        out9 = self.conv_block9(out8)
        out10 = self.conv_block10(out9)
        out11 = self.conv_block11(out10)
        out12 = self.conv_block12(out11)
        out13 = self.conv_block13(out12)
        out14 = self.conv_block14(out13)
        out15 = self.conv_block15(out14)
        out16 = self.conv_block16(out15)
        return out16
    '''
    def decoder_stage(self, x, skip_connections):
        dec_out1 = self.deconv_block1(x)
        #print(f"Shape after deconv_block1: {dec_out1.shape}")  # Debugging line
        #print(f"Before skip conn. dec_out1: {dec_out1.shape}")

        dec_out1 = ME.cat(dec_out1, skip_connections[0])  # Skip connection
        #print(f"Shape after concatenating skip connection: {dec_out1.shape}")  # Debugging line
        #print(f"After skip conn. dec_out1: {dec_out1.shape}")
        dec_out2 = self.deconv_block2(dec_out1)
        dec_out2 = ME.cat(dec_out2, skip_connections[1])  # Skip connection
        dec_out3 = self.deconv_block3(dec_out2)
        dec_out3 = ME.cat(dec_out3, skip_connections[2])  # Skip connection
        dec_out4 = self.deconv_block4(dec_out3)
        dec_out4 = ME.cat(dec_out4, skip_connections[3])  # Skip connection
        print(f"Before skip conn. dec_out4: {dec_out4.shape}")

        dec_out5 = self.deconv_block5(dec_out4)
        dec_out5 = ME.cat(dec_out5, skip_connections[4])  # Skip connection
        #dec_out6 = self.deconv_block6(dec_out5)
        #dec_out6 = ME.cat(dec_out6, skip_connections[5])  # Skip connection
        #dec_out7 = self.deconv_block7(dec_out6)
        #dec_out7 = ME.cat(dec_out7, skip_connections[6])  # Skip connection
        out = self.final_layer(dec_out5)
        return out
        '''
    
    def decoder_stage(self, x, skip_connections):
        dec_out1 = self.deconv_block1(x)
        #print(f"Shape after deconv_block1: {dec_out1.shape}")

        dec_out1 = ME.cat(dec_out1, skip_connections[0])
        #print(f"Shape after deconv_block1 skipp conn: {dec_out1.shape}")

        dec_out2 = self.deconv_block2(dec_out1)
        #print(f"Shape after deconv_block2: {dec_out2.shape}")

        dec_out2 = ME.cat(dec_out2, skip_connections[1])
        #print(f"Shape after deconv_block2 skipp conn: {dec_out2.shape}")

        dec_out3 = self.deconv_block3(dec_out2)
        #print(f"Shape after deconv_block3: {dec_out3.shape}")

        dec_out3 = ME.cat(dec_out3, skip_connections[2])
        #print(f"Shape after deconv_block3 skipp conn: {dec_out3.shape}")

        dec_out4 = self.deconv_block4(dec_out3)
        #print(f"Shape after deconv_block4: {dec_out4.shape}")

        dec_out4 = ME.cat(dec_out4, skip_connections[3])
        #print(f"Shape after deconv_block4 skipp conn: {dec_out4.shape}")

        #dec_out5 = self.deconv_block5(dec_out4)
        #print(f"Shape after deconv_block5: {dec_out5.shape}")

        out = self.final_layer(dec_out4)
        return out


        
        # pruning
        self.pruning = ME.MinkowskiPruning()
        
    def calculate_masked_mse_loss(self, pred_tensor, target_tensor, crit_energy):
        # Extract coordinates and features from the tensors
        pred_coords = pred_tensor.C
        pred_features = pred_tensor.F.squeeze()
        target_coords = target_tensor.C
        target_features = target_tensor.F.squeeze()
        
        #assert len(torch.unique(pred_coords[:, 0])) == len(torch.unique(target_coords[:, 0])), f"Predictions and Targets do not have the same number of tracks. {len(torch.unique(pred_coords[:, 0]))}, {len(torch.unique(target_coords[:, 0]))}"


        # Concatenate the tensors while preserving the batch coordinate
        combined = torch.cat([pred_coords, target_coords], dim=0)
        
        # Get unique coordinates and counts
        unique_coords, counts = combined.unique(dim=0, return_counts=True)

        # Find intersections (coordinates appearing more than once)
        intersection = unique_coords[counts > 1]
        

        # Find indices of intersections in tensor a
        a_expand = pred_coords.unsqueeze(1).expand(-1, intersection.size(0), -1)
        intersection_expand_a = intersection.unsqueeze(0).expand(pred_coords.size(0), -1, -1)
        pred_indices = (a_expand == intersection_expand_a).all(dim=2).any(dim=1).nonzero(as_tuple=True)[0]
        
        # Find indices of intersections in tensor b
        b_expand = target_coords.unsqueeze(1).expand(-1, intersection.size(0), -1)
        intersection_expand_b = intersection.unsqueeze(0).expand(target_coords.size(0), -1, -1)
        target_indices = (b_expand == intersection_expand_b).all(dim=2).any(dim=1).nonzero(as_tuple=True)[0]
        
        #del b_expand, intersection_expand_b, a_expand, intersection_expand_a
        
        assert len(pred_indices) == len(target_indices), f"Prediction and Target indices not the same length. There is a problem. {len(pred_indices)}, {len(target_indices)}" 

        # Gather the features that correspond to the common coordinates
        pred_common_features = pred_features[pred_indices]
        #print("pred common feats", pred_common_features)
        #if torch.isnan(pred_common_features):
        #    print("THERE IS A NAN IN PRED ENERGIES")
        #print(pred_common_features)
        target_common_features = target_features[target_indices]
        #print("target common feats", target_common_features)
        #if torch.isnan(target_common_features):
        #    print("THERE IS A NAN IN TARGET ENERGIES")
        
        #print("t", target_common_features)  

        # Calculate the MSE loss
        loss = crit_energy(pred_common_features, target_common_features)
        #print(loss.item(), type(loss.item()))
        
        # Check if the loss is NaN and if so, set it to 0
        if torch.isnan(loss):
            print("THERE IS A NAN IN LOSS") # NAN is returned if there are no common voxels in prediction and target
            #loss = torch.tensor(0.0, device=loss.device)
        #print(loss)
        return loss
    '''
    def calculate_closest_mse_loss_per_voxel(self, pred_tensor, target_tensor, crit_energy, pred_dec_mask=None, xmin=104, xmax=116, ymin=139, ymax=151, rmse=False, calculate_in_only_inactive=False):
        
        if target_tensor != 0:
        
            # Extract coordinates and features from the tensors
            pred_coords = pred_tensor.C
            pred_features = pred_tensor.F.squeeze()
            target_coords = target_tensor.C
            target_features = target_tensor.F.squeeze()
            
            # use dec pred mask if need be
            if pred_dec_mask is not None:
                pred_coords = pred_coords[pred_dec_mask]
                pred_features = pred_features[pred_dec_mask]

            mask_pred = ((pred_coords[:, 1] >= xmin) & (pred_coords[:, 1] <= xmax) | (pred_coords[:, 2] >= ymin) & (pred_coords[:, 2] <= ymax))
            mask_target = ((target_coords[:, 1] >= xmin) & (target_coords[:, 1] <= xmax) | (target_coords[:, 2] >= ymin) & (target_coords[:, 2] <= ymax))
            
            if calculate_in_only_inactive:
                pass
                #pred_coords = pred_coords[mask_pred]
                #target_coords = target_coords[mask_target]

            # Expand dimensions to calculate distances between all pairs
            pred_coords_expand = pred_coords.float().unsqueeze(1).expand(-1, target_coords.size(0), -1)
            target_coords_expand = target_coords.float().unsqueeze(0).expand(pred_coords.size(0), -1, -1)

            # Calculate squared distances between each pair of voxels
            distances = torch.norm(pred_coords_expand - target_coords_expand, dim=2)

            # Find the index of the closest target voxel for each predicted voxel
            closest_target_indices = torch.argmin(distances, dim=1)

            # Gather the features of the closest target voxels
            closest_target_features = target_features[closest_target_indices]

            # Calculate the MSE loss between predicted features and closest target features
            #print(pred_features.size, closest_target_features.size)


            if rmse:
                loss = torch.sqrt(crit_energy(pred_features, closest_target_features))
            else:
                loss = crit_energy(pred_features, closest_target_features)
       
        else:
            
            loss = torch.tensor(0, dtype=torch.float)

        return loss
        '''

    def calculate_closest_mse_loss_per_voxel(self, pred_tensor, target_tensor, crit_energy, pred_dec_mask=None, xmin=104, xmax=116, ymin=139, ymax=151, rmse=False, calculate_in_only_inactive=False, chunk_size=10):

        if target_tensor != 0:

            # Extract coordinates and features from the tensors
            pred_coords = pred_tensor.C
            pred_features = pred_tensor.F.squeeze()
            target_coords = target_tensor.C
            target_features = target_tensor.F.squeeze()

            # use dec pred mask if need be
            if pred_dec_mask is not None:
                pred_coords = pred_coords[pred_dec_mask]
                pred_features = pred_features[pred_dec_mask]

            mask_pred = ((pred_coords[:, 1] >= xmin) & (pred_coords[:, 1] <= xmax) | (pred_coords[:, 2] >= ymin) & (pred_coords[:, 2] <= ymax))
            mask_target = ((target_coords[:, 1] >= xmin) & (target_coords[:, 1] <= xmax) | (target_coords[:, 2] <= ymax) & (target_coords[:, 2] >= ymin))

            if calculate_in_only_inactive:
                pred_coords = pred_coords[mask_pred]
                target_coords = target_coords[mask_target]

            # Initialize a tensor to store minimum distances
            min_distances = torch.full((pred_coords.size(0),), float('inf'), device=pred_coords.device)
            closest_target_indices = torch.zeros((pred_coords.size(0),), dtype=torch.long, device=pred_coords.device)

            # Process target coordinates in chunks to reduce memory usage
            for start_idx in range(0, target_coords.size(0), chunk_size):
                end_idx = min(start_idx + chunk_size, target_coords.size(0))
                target_coords_chunk = target_coords[start_idx:end_idx].float()

                # Compute distances for the current chunk
                distances_chunk = torch.norm(pred_coords.float().unsqueeze(1) - target_coords_chunk.unsqueeze(0), dim=2)

                # Update the minimum distances and corresponding indices
                min_distances_chunk, min_indices_chunk = torch.min(distances_chunk, dim=1)
                min_distances = torch.min(min_distances, min_distances_chunk)
                closest_target_indices = torch.where(min_distances_chunk < min_distances, min_indices_chunk + start_idx, closest_target_indices)

            # Gather the features of the closest target voxels
            closest_target_features = target_features[closest_target_indices]

            # Calculate the MSE loss between predicted features and closest target features
            if rmse:
                loss = torch.sqrt(crit_energy(pred_features, closest_target_features))
            else:
                loss = crit_energy(pred_features, closest_target_features)

        else:
            loss = torch.tensor(0, dtype=torch.float, device=pred_tensor.device)

        return loss


    
    def combine_sparse_tensors_replace(self, partial_in, pred=None, pred_coords_input=None, pred_feats_input=None, xmin=104, xmax=116, ymin=139, ymax=151, use_mean=False, replace_vals = False):
        """
        Combine two sparse tensors, taking features from partial_in where coordinates overlap.

        Args:
        partial_in (ME.SparseTensor): Sparse tensor with initial coordinates and features.
        pred (ME.SparseTensor): Sparse tensor with predicted coordinates and features.
        xmin (int): Minimum x coordinate to replace features.
        xmax (int): Maximum x coordinate to replace features.
        ymin (int): Minimum y coordinate to replace features.
        ymax (int): Maximum y coordinate to replace features.
        use_mean (bool): If True, replace with the mean of the rest, otherwise replace with -1.

        Returns:
        ME.SparseTensor: Combined sparse tensor.
        """
        # Extract coordinates and features from the input tensors
        # pred has been split into C and F and already passed in as C and F
        partial_in_coords = partial_in.C
        partial_in_features = partial_in.F
        if pred_coords_input is not None:
            pred_coords = pred_coords_input
        else:
            pred_coords = pred.C
        
        if pred_feats_input is not None:
            pred_features = pred_feats_input
        else:
            pred_features = pred.F
        
        #print(partial_in_coords)
        #print(partial_in_features)
        #print(pred_coords)
        #print(pred_features)

        # Concatenate coordinates and features from both tensors
        #print(partial_in_features.size(), pred_features.size())
        combined_coords = torch.cat([partial_in_coords, pred_coords], dim=0)
        pred_features = pred_features.unsqueeze(1)
        #print(partial_in_features.shape)
        #print(pred_features.shape)
        combined_features = torch.cat([partial_in_features, pred_features], dim=0)

        # Get unique coordinates and corresponding indices
        unique_coords, inverse_indices = combined_coords.unique(dim=0, return_inverse=True)

        # Initialize a tensor to store the combined features
        combined_features_tensor = torch.zeros((unique_coords.size(0), combined_features.size(1)), device=combined_features.device)

        # Create a mask to determine if a coordinate is from partial_in or pred
        is_from_partial_in = torch.zeros((unique_coords.size(0),), dtype=torch.bool, device=combined_features.device)
        partial_in_inverse_indices = inverse_indices[:partial_in_coords.size(0)]
        is_from_partial_in[partial_in_inverse_indices] = True

        # Assign features from partial_in to the combined feature tensor
        combined_features_tensor[partial_in_inverse_indices] = partial_in_features

        # Assign features from pred to the combined feature tensor for non-overlapping coordinates
        pred_inverse_indices = inverse_indices[partial_in_coords.size(0):]
        non_overlap_mask = ~is_from_partial_in[pred_inverse_indices]
        combined_features_tensor[pred_inverse_indices[non_overlap_mask]] = pred_features[non_overlap_mask]
        
       
        
        if replace_vals:
            # Mask to identify coordinates within specified x and y ranges
            mask = torch.ones(unique_coords.size(0), dtype=torch.bool, device=combined_features.device)
            if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
                mask = ((unique_coords[:, 1] >= xmin) & (unique_coords[:, 1] <= xmax) | (unique_coords[:, 2] >= ymin) & (unique_coords[:, 2] <= ymax))

            # Compute the replacement value (mean of the rest or zero)
            if use_mean:
                replacement_value = combined_features_tensor[~mask].mean(dim=0, keepdim=True)
            else:
                #replacement_value = torch.zeros((1, combined_features_tensor.size(1)), device=combined_features_tensor.device)
                replacement_value = torch.full((1, combined_features_tensor.size(1)), -1, device=combined_features_tensor.device, dtype=torch.float32)


            # Replace the features for the masked coordinates
            combined_features_tensor[mask] = replacement_value
            
        # track where pred (dec) input voxels exist in unique coordinates and their features in combined_features_tensor
        pred_exist_mask = torch.zeros((unique_coords.size(0),), dtype=torch.bool, device=combined_features.device)
        pred_exist_mask[pred_inverse_indices] = True
        
        # Grab the features for pred input voxels
        pred_features_in_combined = combined_features_tensor[pred_exist_mask]
        
        # Grab actual voxels above pred voxel features
        pred_coords_in_combined = unique_coords[pred_exist_mask]

        # Create the combined sparse tensor
        #combined_tensor = ME.SparseTensor(features=combined_features_tensor, coordinates=unique_coords, coordinate_manager=partial_in.coordinate_manager)
        combined_tensor = ME.SparseTensor(features=combined_features_tensor, coordinates=unique_coords)

        return combined_tensor, pred_exist_mask


    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0]
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target
    
    
    def get_target_new(self, out, target_key, dec_s32_cls_E_temp, dec_mask, target_tensor, crit_energy, kernel_size=1):
        with torch.no_grad():
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0]
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            #print(target_key)
            #print(len(out))
            #print(dec_s32_cls_E_temp)
            #print(kernel_map.items())
            #print(kernel_map)
            #print(len(curr_in))

            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            
            if target_tensor != 0:
                # Calculate the MSE loss for the provided region
                rmse_loss = self.calculate_closest_mse_loss_per_voxel(
                    dec_s32_cls_E_temp, target_tensor, self.crit_energy, dec_mask, 
                    xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
                )
                #print(len(out), len(rmse_loss))

                for k, curr_in in kernel_map.items():
                    curr_indices = curr_in[0].long()
                    #print("here is k", k)
                    #print("here is curr_in", curr_in[0].long())
                    # Only set target to 1 if both conditions are met:
                    # 1. The current logic in get_target.
                    # 2. The calculated MSE loss is less than or equal to the threshold.
                    #print(len(kernel_map), len(rmse_loss), )
                    if rmse_loss[curr_indices].max() <= self.rmse_threshold:
                        target[curr_indices] = 1
            
        #print(target)
        
        # Check if there are any True values
        has_true = torch.any(target) # somthing is wrong with this logic for now (maybe??)
        #print(target)

        #print("Topology and mse loss check returns", has_true)
        return target

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
    
    def forward(self, partial_in, target_key, target_tensor, val_train=False, val_test=False, test=False):
        loss_cls = []
        loss_E = []
        out_cls, targets = [], []

        # doing ME encoding
        enc_s1 = self.enc_block_s1(partial_in)
        enc_s2 = self.enc_block_s1s2(enc_s1)
        enc_s4 = self.enc_block_s2s4(enc_s2)
        enc_s8 = self.enc_block_s4s8(enc_s4)
        enc_s16 = self.enc_block_s8s16(enc_s8)
        enc_s32 = self.enc_block_s16s32(enc_s16)
        enc_s64 = self.enc_block_s32s64(enc_s32)

        # do ME generative decoding, with Element-wise Addition Skip Connection (residual skip connection),
        # then classify through a classification layer. Column 0 will serve as a scoring or existence likihood, column 1 will serve as embedding for energy
        # Decoder 64 -> 32
        dec_s32 = self.dec_block_s64s32(enc_s64) + enc_s32
        dec_s32_cls = self.dec_s32_cls(dec_s32)

        # apply a softplus activation to the energy embedding (this removes negative values, but does not set them directly to 0)
        # combine the known voxels and features (passed in) with the generated voxels and features, taking known voxels and features where the two overlap
        # in new tensor, -1 values will be used as features for voxels that lie in the inactive region (this will be a mask used later to return predicted energy values)
        # return the new tensor and also return a mask pointing to where the generated voxels and features exist in this new tensor (counting the overlaps)
        dec_s32_cls_E_temp1, dec_s32_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s32_cls.C, self.softplus(dec_s32_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        # push the new tensor through its own Unet, to predict energies
        # Energy cnn
        enc_out1_s32_E = self.conv_block1(dec_s32_cls_E_temp1)
        enc_out2_s32_E = self.conv_block2(enc_out1_s32_E)
        enc_out3_s32_E = self.conv_block3(enc_out2_s32_E)
        enc_out4_s32_E = self.conv_block4(enc_out3_s32_E)
        enc_out5_s32_E = self.conv_block5(enc_out4_s32_E)
        enc_out6_s32_E = self.conv_block6(enc_out5_s32_E)
        enc_out7_s32_E = self.conv_block7(enc_out6_s32_E)
        enc_out8_s32_E = self.conv_block8(enc_out7_s32_E)
        enc_out9_s32_E = self.conv_block9(enc_out8_s32_E)
        enc_out10_s32_E = self.conv_block10(enc_out9_s32_E)

        #attention_out_s32_E = self.attention(enc_out10_s32_E) # attention uses too much memory right now. I will try to reimplement it to be less memory intenstive 
        #attention_out_s32_E = self.dropout_attention(attention_out_s32_E)
        
        skip_connections_s32_E = [enc_out8_s32_E, enc_out6_s32_E, enc_out4_s32_E, enc_out2_s32_E]


        #skip_connections_s32_E = [enc_out8_s32_E, enc_out7_s32_E, enc_out6_s32_E, enc_out5_s32_E, enc_out4_s32_E, enc_out3_s32_E, enc_out2_s32_E]

        #dec_s32_cls_E_temp2 = self.decoder_stage(attention_out_s32_E, skip_connections_s32_E)
        dec_s32_cls_E_temp2 = self.decoder_stage(enc_out10_s32_E, skip_connections_s32_E)


        # Energy ends

        # calculate rmse loss between the closest voxels in target and output of energy CNN. There is an option to calculate only for voxels in inactive region as well (not using this option right now because it is not well defined)
        dec_s32_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s32_cls_E_temp2, target_tensor, self.crit_energy, dec_s32_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )
        
        # if some voxels meet both criteria, keep them, else, keep the ones that meet one or the other
        keep_s32 = torch.logical_and(dec_s32_cls.F[:, 0] > 0, dec_s32_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        #print(keep_s32)
        
        #if not keep_s32.any():
        #    print("Using logical or ...(32)")
        #    keep_s32 = torch.logical_or(dec_s32_cls.F[:, 0] > 0, dec_s32_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        if not keep_s32.any():
            print("Using logical or ...(32)")
            keep_s32 = torch.logical_or(dec_s32_cls.F[:, 0] > 0, dec_s32_cls_E_temp_rmse <= self.rmse_threshold).squeeze()      
        else:
            print("Using logical and ...(32)")
        if self.training:
            print("RSME", dec_s32_cls_E_temp_rmse)
            print("CLS", dec_s32_cls.F[:, 0])  

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s32, target_key, dec_s32_cls_E_temp2, dec_s32_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s32_cls.F[:, 0].squeeze(), target_cls.type(dec_s32_cls.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s32_cls_E_temp_rmse.mean())
        if self.training:
            keep_s32 += target_cls

        dec_s32 = self.pruning(dec_s32, keep_s32)

        del dec_s32_cls, target_cls, keep_s32, skip_connections_s32_E
        gc.collect()
        torch.cuda.empty_cache()

        # Decoder 32 -> 16
        dec_s16 = self.dec_block_s32s16(dec_s32) + enc_s16
        dec_s16_cls = self.dec_s16_cls(dec_s16)

        dec_s16_cls_E_temp1, dec_s16_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s16_cls.C, self.softplus(dec_s16_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s16_E = self.conv_block1(dec_s16_cls_E_temp1)
        enc_out2_s16_E = self.conv_block2(enc_out1_s16_E)
        enc_out3_s16_E = self.conv_block3(enc_out2_s16_E)
        enc_out4_s16_E = self.conv_block4(enc_out3_s16_E)
        enc_out5_s16_E = self.conv_block5(enc_out4_s16_E)
        enc_out6_s16_E = self.conv_block6(enc_out5_s16_E)
        enc_out7_s16_E = self.conv_block7(enc_out6_s16_E)
        enc_out8_s16_E = self.conv_block8(enc_out7_s16_E)
        enc_out9_s16_E = self.conv_block9(enc_out8_s16_E)
        enc_out10_s16_E = self.conv_block10(enc_out9_s16_E)

        #attention_out_s16_E = self.attention(enc_out10_s16_E)
        #attention_out_s16_E = self.dropout_attention(attention_out_s16_E)

        skip_connections_s16_E = [enc_out8_s16_E, enc_out6_s16_E, enc_out4_s16_E, enc_out2_s16_E]

        #dec_s16_cls_E_temp2 = self.decoder_stage(attention_out_s16_E, skip_connections_s16_E)
        dec_s16_cls_E_temp2 = self.decoder_stage(enc_out10_s16_E, skip_connections_s16_E)


        dec_s16_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s16_cls_E_temp2, target_tensor, self.crit_energy, dec_s16_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )
        
        
        keep_s16 = torch.logical_and(dec_s16_cls.F[:, 0] > 0, dec_s16_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        
        if not keep_s16.any():
            print("Using logical or ...(16)")
            keep_s16 = torch.logical_or(dec_s16_cls.F[:, 0] > 0, dec_s16_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        else:
            print("Using logical and ...(16)")

            

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s16, target_key, dec_s16_cls_E_temp2, dec_s16_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s16_cls.F[:, 0].squeeze(), target_cls.type(dec_s16_cls.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s16_cls_E_temp_rmse.mean())
        if self.training:
            keep_s16 += target_cls

        dec_s16 = self.pruning(dec_s16, keep_s16)

        del dec_s16_cls, target_cls, keep_s16, skip_connections_s16_E
        gc.collect()
        torch.cuda.empty_cache()

        # Decoder 16 -> 8
        dec_s8 = self.dec_block_s16s8(dec_s16) + enc_s8
        dec_s8_cls = self.dec_s8_cls(dec_s8)

        dec_s8_cls_E_temp1, dec_s8_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s8_cls.C, self.softplus(dec_s8_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s8_E = self.conv_block1(dec_s8_cls_E_temp1)
        enc_out2_s8_E = self.conv_block2(enc_out1_s8_E)
        enc_out3_s8_E = self.conv_block3(enc_out2_s8_E)
        enc_out4_s8_E = self.conv_block4(enc_out3_s8_E)
        enc_out5_s8_E = self.conv_block5(enc_out4_s8_E)
        enc_out6_s8_E = self.conv_block6(enc_out5_s8_E)
        enc_out7_s8_E = self.conv_block7(enc_out6_s8_E)
        enc_out8_s8_E = self.conv_block8(enc_out7_s8_E)
        enc_out9_s8_E = self.conv_block9(enc_out8_s8_E)
        enc_out10_s8_E = self.conv_block10(enc_out9_s8_E)

        #attention_out_s8_E = self.attention(enc_out10_s8_E)
        #attention_out_s8_E = self.dropout_attention(attention_out_s8_E)

        skip_connections_s8_E = [enc_out8_s8_E, enc_out6_s8_E, enc_out4_s8_E, enc_out2_s8_E]

        dec_s8_cls_E_temp2 = self.decoder_stage(enc_out10_s8_E, skip_connections_s8_E)

        dec_s8_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s8_cls_E_temp2, target_tensor, self.crit_energy, dec_s8_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )

        keep_s8 = torch.logical_and(dec_s8_cls.F[:, 0] > 0, dec_s8_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        
        if not keep_s8.any():
            print("Using logical or ...(8)")
            keep_s8 = torch.logical_or(dec_s8_cls.F[:, 0] > 0, dec_s8_cls_E_temp_rmse <= self.rmse_threshold).squeeze()

        else:
            print("Using logical and ...(8)")
            

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s8, target_key, dec_s8_cls_E_temp2, dec_s8_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s8_cls.F[:, 0].squeeze(), target_cls.type(dec_s8_cls.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s8_cls_E_temp_rmse.mean())
        if self.training:
            keep_s8 += target_cls

        dec_s8 = self.pruning(dec_s8, keep_s8)

        del dec_s8_cls, target_cls, keep_s8, skip_connections_s8_E
        gc.collect()
        torch.cuda.empty_cache()

        # Decoder 8 -> 4
        dec_s4 = self.dec_block_s8s4(dec_s8) + enc_s4
        dec_s4_cls = self.dec_s4_cls(dec_s4)

        dec_s4_cls_E_temp1, dec_s4_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s4_cls.C, self.softplus(dec_s4_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s4_E = self.conv_block1(dec_s4_cls_E_temp1)
        enc_out2_s4_E = self.conv_block2(enc_out1_s4_E)
        enc_out3_s4_E = self.conv_block3(enc_out2_s4_E)
        enc_out4_s4_E = self.conv_block4(enc_out3_s4_E)
        enc_out5_s4_E = self.conv_block5(enc_out4_s4_E)
        enc_out6_s4_E = self.conv_block6(enc_out5_s4_E)
        enc_out7_s4_E = self.conv_block7(enc_out6_s4_E)
        enc_out8_s4_E = self.conv_block8(enc_out7_s4_E)
        enc_out9_s4_E = self.conv_block9(enc_out8_s4_E)
        enc_out10_s4_E = self.conv_block10(enc_out9_s4_E)

        #attention_out_s4_E = self.attention(enc_out10_s4_E)
        #attention_out_s4_E = self.dropout_attention(attention_out_s4_E)

        skip_connections_s4_E = [enc_out8_s4_E, enc_out6_s4_E, enc_out4_s4_E, enc_out2_s4_E]

        dec_s4_cls_E_temp2 = self.decoder_stage(enc_out10_s4_E, skip_connections_s4_E)

        dec_s4_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s4_cls_E_temp2, target_tensor, self.crit_energy, dec_s4_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )

        keep_s4 = torch.logical_and(dec_s4_cls.F[:, 0] > 0, dec_s4_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        
        if not keep_s4.any():
            print("Using logical or ...(4)")
            keep_s4 = torch.logical_or(dec_s4_cls.F[:, 0] > 0, dec_s4_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        else:
            print("Using logical and ...(4)")
            

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s4, target_key, dec_s4_cls_E_temp2, dec_s4_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s4_cls.F[:, 0].squeeze(), target_cls.type(dec_s4_cls.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s4_cls_E_temp_rmse.mean())
        if self.training:
            keep_s4 += target_cls

        dec_s4 = self.pruning(dec_s4, keep_s4)

        del dec_s4_cls, target_cls, keep_s4, skip_connections_s4_E
        gc.collect()
        torch.cuda.empty_cache()

        # Decoder 4 -> 2
        dec_s2 = self.dec_block_s4s2(dec_s4) + enc_s2
        dec_s2_cls = self.dec_s2_cls(dec_s2)

        dec_s2_cls_E_temp1, dec_s2_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s2_cls.C, self.softplus(dec_s2_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s2_E = self.conv_block1(dec_s2_cls_E_temp1)
        enc_out2_s2_E = self.conv_block2(enc_out1_s2_E)
        enc_out3_s2_E = self.conv_block3(enc_out2_s2_E)
        enc_out4_s2_E = self.conv_block4(enc_out3_s2_E)
        enc_out5_s2_E = self.conv_block5(enc_out4_s2_E)
        enc_out6_s2_E = self.conv_block6(enc_out5_s2_E)
        enc_out7_s2_E = self.conv_block7(enc_out6_s2_E)
        enc_out8_s2_E = self.conv_block8(enc_out7_s2_E)
        enc_out9_s2_E = self.conv_block9(enc_out8_s2_E)
        enc_out10_s2_E = self.conv_block10(enc_out9_s2_E)

        #attention_out_s2_E = self.attention(enc_out10_s2_E)
        #attention_out_s2_E = self.dropout_attention(attention_out_s2_E)

        skip_connections_s2_E = [enc_out8_s2_E, enc_out6_s2_E, enc_out4_s2_E, enc_out2_s2_E]

        dec_s2_cls_E_temp2 = self.decoder_stage(enc_out10_s2_E, skip_connections_s2_E)

        dec_s2_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s2_cls_E_temp2, target_tensor, self.crit_energy, dec_s2_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )

        keep_s2 = torch.logical_and(dec_s2_cls.F[:, 0] > 0, dec_s2_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        
        if not keep_s2.any():
            print("Using logical or ...(2)")
            keep_s2 = torch.logical_or(dec_s2_cls.F[:, 0] > 0, dec_s2_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        else:
            print("Using logical and ...(2)")
            

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s2, target_key, dec_s2_cls_E_temp2, dec_s2_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s2_cls.F[:, 0].squeeze(), target_cls.type(dec_s2_cls.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s2_cls_E_temp_rmse.mean())
        if self.training:
            keep_s2 += target_cls

        dec_s2 = self.pruning(dec_s2, keep_s2)

        del dec_s2_cls, target_cls, keep_s2, skip_connections_s2_E
        gc.collect()
        torch.cuda.empty_cache()

        # Decoder 2 -> 1
        dec_s1 = self.dec_block_s2s1(dec_s2) + enc_s1
        dec_s1_cls = self.dec_s1_cls(dec_s1)

        dec_s1_cls_E_temp1, dec_s1_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s1_cls.C, self.softplus(dec_s1_cls.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s1_E = self.conv_block1(dec_s1_cls_E_temp1)
        enc_out2_s1_E = self.conv_block2(enc_out1_s1_E)
        enc_out3_s1_E = self.conv_block3(enc_out2_s1_E)
        enc_out4_s1_E = self.conv_block4(enc_out3_s1_E)
        enc_out5_s1_E = self.conv_block5(enc_out4_s1_E)
        enc_out6_s1_E = self.conv_block6(enc_out5_s1_E)
        enc_out7_s1_E = self.conv_block7(enc_out6_s1_E)
        enc_out8_s1_E = self.conv_block8(enc_out7_s1_E)
        enc_out9_s1_E = self.conv_block9(enc_out8_s1_E)
        enc_out10_s1_E = self.conv_block10(enc_out9_s1_E)

        #attention_out_s1_E = self.attention(enc_out10_s1_E)
        #attention_out_s1_E = self.dropout_attention(attention_out_s1_E)

        skip_connections_s1_E = [enc_out8_s1_E, enc_out6_s1_E, enc_out4_s1_E, enc_out2_s1_E]

        dec_s1_cls_E_temp2 = self.decoder_stage(enc_out10_s1_E, skip_connections_s1_E)

        dec_s1_cls_E_temp_rmse = self.calculate_closest_mse_loss_per_voxel(
            dec_s1_cls_E_temp2, target_tensor, self.crit_energy, dec_s1_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )

        keep_s1 = torch.logical_and(dec_s1_cls.F[:, 0] > 0, dec_s1_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        if not keep_s1.any():
            print("Using logical or ...(1-1st)")
            keep_s1 = torch.logical_or(dec_s1_cls.F[:, 0] > 0, dec_s1_cls_E_temp_rmse <= self.rmse_threshold).squeeze()
        else:
            print("Using logical and ...(1-1st)")
    

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s1, target_key, dec_s1_cls_E_temp2, dec_s1_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s1.F[:, 0].squeeze(), target_cls.type(dec_s1.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s1_cls_E_temp_rmse.mean())
        if self.training:
            keep_s1 += target_cls

        dec_s1 = self.pruning(dec_s1, keep_s1)

        del dec_s1_cls, target_cls, keep_s1, skip_connections_s1_E
        gc.collect()
        torch.cuda.empty_cache()

        # LAST FINAL
        dec_s1 = self.final_out_cls(dec_s1)

        dec_s1_final_cls_E_temp1, dec_s1_final_cls_E_temp1_mask = self.combine_sparse_tensors_replace(
            partial_in, None, dec_s1.C, self.softplus(dec_s1.F[:, 1]).squeeze(),
            xmin=104, xmax=116, ymin=139, ymax=151,
            use_mean=False, replace_vals=True
        )

        enc_out1_s1_E_final = self.conv_block1(dec_s1_final_cls_E_temp1)
        enc_out2_s1_E_final = self.conv_block2(enc_out1_s1_E_final)
        enc_out3_s1_E_final = self.conv_block3(enc_out2_s1_E_final)
        enc_out4_s1_E_final = self.conv_block4(enc_out3_s1_E_final)
        enc_out5_s1_E_final = self.conv_block5(enc_out4_s1_E_final)
        enc_out6_s1_E_final = self.conv_block6(enc_out5_s1_E_final)
        enc_out7_s1_E_final = self.conv_block7(enc_out6_s1_E_final)
        enc_out8_s1_E_final = self.conv_block8(enc_out7_s1_E_final)
        enc_out9_s1_E_final = self.conv_block9(enc_out8_s1_E_final)
        enc_out10_s1_E_final = self.conv_block10(enc_out9_s1_E_final)

        #attention_out_s1_E_final = self.attention(enc_out10_s1_E_final)
        #attention_out_s1_E_final = self.dropout_attention(attention_out_s1_E_final)

        skip_connections_s1_E_final = [enc_out8_s1_E_final, enc_out6_s1_E_final, enc_out4_s1_E_final, enc_out2_s1_E_final]
        dec_s1_cls_E_temp2_final = self.decoder_stage(enc_out10_s1_E_final, skip_connections_s1_E_final)

        dec_s1_cls_E_temp_rmse_final = self.calculate_closest_mse_loss_per_voxel(
            dec_s1_cls_E_temp2_final, target_tensor, self.crit_energy, dec_s1_final_cls_E_temp1_mask,
            xmin=104, xmax=116, ymin=139, ymax=151, rmse=True, calculate_in_only_inactive=False
        )

        keep_s1 = torch.logical_and(dec_s1.F[:, 0] > 0, dec_s1_cls_E_temp_rmse_final <= self.rmse_threshold).squeeze()
        if not keep_s1.any():
            print("Using logical or ...(1-2nd)")
            keep_s1 = torch.logical_or(dec_s1.F[:, 0] > 0, dec_s1_cls_E_temp_rmse_final <= self.rmse_threshold).squeeze()
        else:
            print("Using logical and ...(1-2nd)")

            

        target_cls = 0
        if target_key != 0:
            target_cls = self.get_target_new(dec_s1, target_key, dec_s1_cls_E_temp2_final, dec_s1_final_cls_E_temp1_mask, target_tensor, self.crit_energy)
            loss_cls.append(self.crit_cls(dec_s1.F[:, 0].squeeze(), target_cls.type(dec_s1.F[:, 0].dtype).to(device)))
            loss_E.append(dec_s1_cls_E_temp_rmse_final.mean())
        if self.training:
            keep_s1 += target_cls

        dec_s1 = self.pruning(dec_s1, keep_s1)
        del target_cls, keep_s1, skip_connections_s1_E_final

        if loss_cls:
            final_batch_cls_loss = torch.mean(torch.stack(loss_cls))
        else:
            final_batch_cls_loss = torch.tensor(0, dtype=torch.float)

        if loss_E:
            final_energy_loss = torch.mean(torch.stack(loss_E))
        else:
            final_energy_loss = torch.tensor(0, dtype=torch.float)

        out_s1_E_final = dec_s1_cls_E_temp2_final

        return out_s1_E_final, targets, dec_s1, final_batch_cls_loss + final_energy_loss, final_batch_cls_loss, final_energy_loss
