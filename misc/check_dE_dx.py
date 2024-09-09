import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def is_michel_track(voxels):
    # trying to find michel electrons at the end of muon tracks. (this is mostly unused/uneeded for now)
    electron_voxels = voxels[voxels[:, 4] == 11]
    if len(electron_voxels) > 10:
        return True
    return False

def find_bragg_peak(track):
    # A high energy threshold to identify potential Bragg peak candidates; might change this to 2*np.mean(track[:, 3])
    high_energy_threshold = 0.98 * np.max(track[:, 3])
    
    # Find voxels with energy above the threshold, must be muon voxels!!
    high_energy_voxels = track[(track[:, 3] >= high_energy_threshold) & (track[:, 4] == 13)]
    
    # Sort the high-energy voxels by energy in descending order
    high_energy_voxels = high_energy_voxels[np.argsort(high_energy_voxels[:, 3])[::-1]]
    
    # Identify the Bragg peak, look for a voxel in the group above that is surrounded by other high-energy voxels
    # this is probably not needed
    bragg_peak = None
    for voxel in high_energy_voxels:
        neighbors = high_energy_voxels[
            np.all(np.abs(high_energy_voxels[:, :3] - voxel[:3]) <= 1, axis=1)
        ]
        if len(neighbors) > 3:  # heuristic: more than 3 neighbors within a small distance
            bragg_peak = voxel
            break
    
    # Default to the highest energy voxel if no "cluster" is found
    # return bragg_peak if bragg_peak is not None else high_energy_voxels[0] #ignore this
    return high_energy_voxels[0]


def calculate_directionality_along_track(track):
    # Ignore small clusters of PDG 11 (electrons) along the muon track
    muon_track = track[track[:, 4] == 13]

    if len(muon_track) == 0:
        print("returning None??")
        return None, None, None  # No muon track found

    # Identify the Bragg peak
    bragg_peak = find_bragg_peak(muon_track)

    # Get the index of the Bragg peak in the muon track
    bragg_peak_index = np.where((muon_track[:, :3] == bragg_peak[:3]).all(axis=1))[0][0]

    # Calculate the sequence of direction vectors from the start of the track to the Bragg peak
    # this may need to be slightly more sophisticated, though I only care about direction here, so maybe this is ok
    direction_vectors = np.diff(muon_track[:bragg_peak_index + 1, :3], axis=0)
    
    # The start voxel is the first voxel in the track
    start_voxel = muon_track[0, :3]

    return direction_vectors, start_voxel, bragg_peak



def calculate_residual_range_along_track(track, bragg_peak, voxel_size_cm=0.38):
    # Find the index of the Bragg peak in the track
    bragg_peak_index = np.where((track[:, :3] == bragg_peak[:3]).all(axis=1))[0][0]
    
    # Initialize residual range array
    residual_ranges = np.zeros(len(track))
    
    # Set to keep track of unique residual ranges
    unique_voxel_positions = set()

    # Start traversing from the Bragg peak along the track
    total_distance = 0
    current_index = bragg_peak_index
    visited_voxels = {current_index}
    
    # Define the sorted track 
    sorted_track_indices = [current_index]
    
    # start with bragg peak as the current voxel. Go through every other voxel (those not in visited_voxels), calculate the
    # Euclidean distance between the current voxel and every other voxel. Take the voxel with the minimum distance to be the "next"
    # voxel along the track. The total distance is then updated with the min_distance that is calculated. The "next" voxel is recorded
    while len(visited_voxels) < len(track):
        # Find the nearest neighboring voxel that hasn't been visited
        min_distance = float('inf')
        next_index = None
        for i in range(len(track)):
            if i not in visited_voxels:
                distance = np.linalg.norm(track[current_index, :3] - track[i, :3])
                if distance < min_distance:
                    min_distance = distance
                    next_index = i
        
        if next_index is not None:
            # Update the total distance along the track
            total_distance += min_distance
            
            # Add the next voxel to the sorted track and visited voxels set
            sorted_track_indices.append(next_index)
            visited_voxels.add(next_index)
            
            # Update the current index to move along the track
            current_index = next_index

    # Sort the track by actual distance along the track!
    sorted_track = track[sorted_track_indices]
    
    # Calculate the residual range along the sorted track
    total_distance = 0
    for i in range(1, len(sorted_track)):
        distance_step = np.linalg.norm(sorted_track[i, :3] - sorted_track[i-1, :3])
        total_distance += distance_step
        residual_range = total_distance * voxel_size_cm  # Convert to cm
        
        # Check if the voxel position is unique, set a "residual range"
        voxel_pos = tuple(sorted_track[i, :3])
        if voxel_pos not in unique_voxel_positions:
            residual_ranges[sorted_track_indices[i]] = residual_range  # Map back to original indices
            unique_voxel_positions.add(voxel_pos)

    return residual_ranges


def grab_and_plot_dEdx_vs_residual_range(h5file_path, max_tracks=1000000):
    residual_ranges_all = []
    dEdx_all = []
    
    with h5py.File(h5file_path, 'r') as f:
        dataset_key = 'images'
        #dataset_key = 'images' if 'fullImage' in h5file_path else 'point_clouds'

        images_data = f[dataset_key]
        
        num_images = len(np.unique(images_data['imageInd']))
        count1 = 0

        #for image_ind in range(min(max_tracks, num_images)):
        for image_ind in np.unique(images_data['imageInd']):
            #if count1 >= 10:
            #    break
            image_indices = images_data['imageInd'] == image_ind
            vox_data = images_data[image_indices]
            #if not np.all(np.isin(vox_data['voxPID'], [13, 11])):
            #    continue
            
            # Extract muon-like voxels
            track = np.vstack((vox_data['voxx'], vox_data['voxy'], vox_data['voxz'], vox_data['voxdE'], vox_data['voxPID'])).T
            muon_track = track[track[:, 4] == 13]  # Ignore electron voxels
            
            #if len(muon_track) == 0:
            #    continue

            # Find the Bragg peak
            direction_vectors, start_voxel, bragg_peak = calculate_directionality_along_track(muon_track)
            if direction_vectors is None:
                continue
            
            # Calculate residual range
            #residual_ranges = calculate_residual_range(muon_track, bragg_peak)
            residual_ranges = calculate_residual_range_along_track(muon_track, bragg_peak)
            
            # Calculate dE/dx
            dEdx = muon_track[:, 3] / 0.38  # dE/dx = energy deposition per cm
            #dEdx = muon_track[:, 3] / 0.1  # dE/dx = energy deposition per cm

            # finalize data for 2d hist plotting
            residual_ranges_all.extend(residual_ranges)
            dEdx_all.extend(dEdx)
            
            if count1 % 50 == 0:
                print(f'Processed track {count1}')
            count1 += 1

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist2d(residual_ranges_all, dEdx_all, bins=(1500, 150), cmap='viridis', norm=plt.Normalize(), cmin=1) #GnBu
    #plt.hist2d(residual_ranges_all, dEdx_all, bins=(100, 1000), cmap='viridis', norm=plt.Normalize(), cmin=1) #GnBu

    #plt.hist2d(residual_ranges_all, dEdx_all, bins=(1000, 1000), cmap='inferno', norm=LogNorm(), cmin=1)

    
    
    #plt.hist2d(residual_ranges_all, dEdx_all, bins=(100, 100), cmap='inferno', cmin=1)
    #plt.hist2d(residual_ranges_all, dEdx_all, bins=(1000, 1000), cmap='inferno', norm=LogNorm(), cmin=1)

    plt.colorbar(label='Counts')
    plt.xlabel('Residual Range [cm]')
    plt.ylabel('dE/dx [MeV/cm]')
    plt.ylim(0, 15)  
    plt.xlim(0, 150)
    plt.title('dE/dx vs. Residual Range for Muon-Like Tracks')
    plt.grid(True)
    plt.show()
  
grab_and_plot_dEdx_vs_residual_range(r"/path/to/h5_file.h5")
