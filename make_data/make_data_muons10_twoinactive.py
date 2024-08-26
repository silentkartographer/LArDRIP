import h5py
import numpy as np
import math
import copy

def find_smallest_track_center_and_volume(tracks):
    smallest_volume = np.inf
    smallest_track_center = None

    for track in tracks:
        if track.size == 0:  # Skip empty tracks
            continue

        min_coords = np.min(track, axis=0)
        max_coords = np.max(track, axis=0)
        volume = np.prod(max_coords - min_coords)
        if volume < smallest_volume:
            smallest_volume = volume
            smallest_track_center = (max_coords + min_coords) / 2  # Center of the track

    if smallest_track_center is None:
        raise ValueError("All tracks are empty or no tracks provided.")

    return smallest_track_center, smallest_volume

def create_dense_grid(center, grid_size):
    """Creates a dense grid of coordinates centered around a given point with specified range and step=1."""
    x_range = np.arange(int(center[0]) - grid_size[0] // 2, int(center[0]) + grid_size[0] // 2 + 1)
    y_range = np.arange(int(center[1]) - grid_size[1] // 2, int(center[1]) + grid_size[1] // 2 + 1)
    z_range = np.arange(int(center[2]) - grid_size[2] // 2, int(center[2]) + grid_size[2] // 2 + 1)
    return np.array(np.meshgrid(x_range, y_range, z_range, indexing='ij')).reshape(3, -1).T

def normalize_tracks(tracks, min_energy, max_energy):
    all_tracks = np.vstack(tracks)
    
    # Min max normalization
    if max_energy > min_energy:
        all_tracks[:, 3] = (all_tracks[:, 3] - min_energy) / (max_energy - min_energy)
    else:
        print("What happened??")
        return 0
    
    # Split the array back into the original list of tracks
    split_indices = np.cumsum([len(track) for track in tracks[:-1]])
    normalized_tracks = np.split(all_tracks, split_indices)
    
    return normalized_tracks
    
    

def grab_particle_tracks(h5file_path, box_size, grid_size):
    tracks = []
    energies = []
    with h5py.File(h5file_path, 'r') as f:
        dataset_key = 'images' if 'fullImage' in h5file_path else 'point_clouds'
        images_data = f[dataset_key]
        
        num_images = len(np.unique(images_data['imageInd']))
        count1 =  0

        for image_ind in range(min(100000000000, num_images)):
            #if len(tracks) >= 20:
            #    break
            image_indices = images_data['imageInd'] == image_ind
            vox_data = images_data[image_indices]
            track = np.vstack((vox_data['voxx'], vox_data['voxy'], vox_data['voxz'], vox_data['voxdE'])).T
            
            # Check for duplicate coordinates and aggregate energy values
            unique_coords, indices, inverse_indices = np.unique(track[:, :3], axis=0, return_index=True, return_inverse=True)
            aggregated_energy = np.zeros((len(unique_coords), 4))
            aggregated_energy[:, :3] = unique_coords
            
            for i in range(len(track)):
                aggregated_energy[inverse_indices[i], 3] += track[i, 3]

            if len(aggregated_energy) < len(track):
                print(f"Duplicate coordinates found and aggregated in track {count1}")

            tracks.append(aggregated_energy)
            energies.append(aggregated_energy[:, 3])
            print(f'making initial track {count1}')
            count1 += 1

    
    
    smallest_track_center, _ = find_smallest_track_center_and_volume(tracks)
    
    flattened_energies = np.hstack(energies)
    min_energy, max_energy = np.min(flattened_energies), np.max(flattened_energies)

    
    tracks = normalize_tracks(tracks, min_energy, max_energy)
    
    # feats = 1
    input_tracks = []
    target_tracks = []
    
    # feats = energy values
    input_tracks_energies = []
    target_tracks_energies = []
    
    # feats = 1
    input_coords_list = []
    input_feats_list = []
    target_feats_list = []
    target_coords_list = []
    
    # feats = energy values
    input_coords_list_energies = []
    input_feats_list_energies = []
    target_feats_list_energies = []
    target_coords_list_energies = []
    
    
    
    count2 = 0
    for track in tracks:
        xmin, xmax = smallest_track_center[0] - box_size[0] / 2, smallest_track_center[0] + box_size[0] / 2
        print(xmin, xmax)
        ymin, ymax = 140, 150
        inside = (track[:, 0] >= xmin) & (track[:, 0] <= xmax) | (track[:, 1] >= ymin) & (track[:, 1] <= ymax)
        missing_region = track[inside]
        before_region = track[~inside & ((track[:, 0] < xmin) | (track[:, 1] < ymin))]
        after_region = track[~inside & ((track[:, 0] > xmax) | (track[:, 1] > ymax))]
        


        if missing_region.size == 0 or before_region.size == 0 or after_region.size == 0:
            continue
            
        # Calculate the true center from the last point before and first point after the missing region
        #true_center = (before_region[-1, :3] + after_region[0, :3]) / 2

        # Generate dense grid
        #grid = create_dense_grid(true_center, grid_size)
                  
        input_track = np.vstack([before_region, after_region])
        #input_track = track[~inside]
        target_track = track
        input_track_energy = copy.deepcopy(track)
        mask = ((input_track_energy[:, 0] >= xmin) & (input_track_energy[:, 0] <= xmax)) | ((input_track_energy[:, 1] >= ymin) & (input_track_energy[:, 1] <= ymax))
        input_track_energy = np.vstack([before_region, after_region])
        #input_track_energy[mask, 3] = 0.00001
        target_track_energy = copy.deepcopy(track)
        input_track[:, 3] = 1
        target_track[:, 3] = 1
        '''
        # Create a mask for the grid region
        grid_mask = np.zeros(len(input_track), dtype=bool)
        start_idx = len(before_region) + len(after_region)
        end_idx = start_idx + len(grid)
        grid_mask[start_idx:end_idx] = True
        '''
        # feats = 1
        input_tracks.append(input_track)
        target_tracks.append(target_track)
        
        # feats = energy values
        input_tracks_energies.append(input_track_energy)
        target_tracks_energies.append(target_track_energy)
       
        # feats = 1
        input_coords = input_track[:, :3]
        input_feats = input_track[:, 3]
        target_feats = target_track[:, 3]
        target_coords = target_track[:, :3]
        
        input_coords_list.append(input_coords)
        input_feats_list.append(input_feats)
        target_feats_list.append(target_feats)
        target_coords_list.append(target_coords)
        
        # feats = energy values
        input_coords_energy = input_track_energy[:, :3]
        input_feats_energy = input_track_energy[:, 3]
        target_feats_energy = target_track_energy[:, 3]
        target_coords_energy = target_track_energy[:, :3]
        
        input_coords_list_energies.append(input_coords_energy)
        input_feats_list_energies.append(input_feats_energy)
        target_feats_list_energies.append(target_feats_energy)
        target_coords_list_energies.append(target_coords_energy)
        
        
        
        
        print(f'making completed tracks {count2}') 
        count2 += 1
    
    print(f"len input_coords {len(input_coords_list)}, len input_feats {len(input_feats_list)}, len target_feats {len(target_feats_list)}")
    print(f"len input_coords {len(input_coords_list_energies)}, len input_feats {len(input_feats_list_energies)}, len target_feats {len(target_feats_list_energies)}")

    return input_tracks, target_tracks, input_coords_list, input_feats_list, target_coords_list, target_feats_list, input_tracks_energies, target_tracks_energies, input_coords_list_energies, input_feats_list_energies, target_coords_list_energies, target_feats_list_energies


file_path = r"/path_to/prepped_projections_224px_muononly.h5"
missing_range = 10
box_size = (missing_range, 1000, 1000)
grid_size = (missing_range, 25, 25)
input_tracks, target_tracks, input_coords, input_feats, target_coords, target_feats, input_tracks_energies, target_tracks_energies, input_coords_energies, input_feats_energies, target_coords_energies, target_feats_energies = grab_particle_tracks(file_path, box_size, grid_size)

if file_path == r"/path_to/prepped_projections_224px_muononly.h5":
    
    np.savez(r'/path_to_save/input_tracks.npz', *input_tracks)
    np.savez(r'/path_to_save/target_tracks.npz', *target_tracks)
    np.savez(r'/path_to_save/input_coords.npz', *input_coords)
    np.savez(r'/path_to_save/input_feats.npz', *input_feats)
    np.savez(r'/path_to_save/target_coords.npz', *target_coords)
    np.savez(r'/path_to_save/target_feats.npz', *target_feats)
    
    
    
    np.savez(r'/path_to_save/input_tracks_energies.npz', *input_tracks_energies)
    np.savez(r'/path_to_save/target_tracks_energies.npz', *target_tracks_energies)
    np.savez(r'/path_to_save/input_coords_energies.npz', *input_coords_energies)
    np.savez(r'/path_to_save/input_feats_energies.npz', *input_feats_energies)
    np.savez(r'/path_to_save/target_coords_energies.npz', *target_coords_energies)
    np.savez(r'/path_to_save/target_feats_energies.npz', *target_feats_energies)
    #np.savez(r'/not_used!!', *grid_masks)
