import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm.auto import trange
import time

import knotgrowth.calculationfunctions as calc

def simulation_loop(current_grid, num_labels, grid_size, penalty_radius, num_iterations, sigma, connectivity_padding, mask_penalty, region_history, volume_conservation):
                
    # visualize_3d_slices(calc.boundary_of_grid(current_grid), 0, num_labels + 1, view=(10,10), figsize=(15,15))
    print(f"Euler characteristic: {calc.compute_surface_euler_characteristic(current_grid, background_label=1)}")

    target_volumes = calc.calculate_3d_volumes(current_grid, num_labels)

    # seed regions code
    seed_masks = {label: np.zeros_like(current_grid, dtype=bool) for label in range(2, num_labels + 1)}

    for label in range(2, num_labels + 1): # This for loop takes 111 seconds (distance_transform_edt takes 0.01 sec and is run approx num_labels*len(coords) times)
        coords = np.argwhere(current_grid == label)
        if len(coords) > 0:
            # num = 40 means there are 40 seed points at each cell region (just choose arbitrarily large)
            selected_indices = coords[np.linspace(0, len(coords) - 1, num=200, dtype=int)]
            for z0, y0, x0 in selected_indices:
                mask = np.zeros_like(current_grid, dtype=bool)
                mask[z0, y0, x0] = True
                dist_map = distance_transform_edt(~mask)
                seed_masks[label] |= (dist_map <= penalty_radius)

    growth_coeff = 2 # Overall scaling of dt and volume_growth_rate. choose between (1,3)

    dt = 0.4*growth_coeff
    volume_growth_rate = 20*growth_coeff

    grid_shape = (grid_size,grid_size,grid_size)
    next_grid = np.empty(grid_shape) 

    for iter_num in trange(num_iterations, desc='simulation loop'):

        if target_volumes[5] > 170: #115 # smaller grid

            
            dt = 0.8*growth_coeff #3.9
            volume_growth_rate = 40*growth_coeff

        if target_volumes[5] > 530: #mid
            
            
            dt =1.6*growth_coeff #4.37   # try 7 8 or 5.5
            volume_growth_rate = 80*growth_coeff

        if target_volumes[5] > 930: #large
            
            
            dt=1.6*growth_coeff  
            volume_growth_rate = 80*growth_coeff
        
        if target_volumes[5] > 4200:# was 2600
            break

        # Update target volumes
        for label_id in range(1, num_labels + 1):
            if label_id == 1:
                target_volumes[label_id] = target_volumes[label_id] - volume_growth_rate * (num_labels - 1)
            else:
                target_volumes[label_id] = target_volumes[label_id] + volume_growth_rate

        # Compute psi fields
        psies = calc.psi_3d_optimized(current_grid, sigma, dt) # 1.35 sec

        # Apply connectivity preservation
        for lbl in range(1, num_labels + 1): # 0.18 sec
            dilated = calc.dilate_boundary_3d(current_grid, np.int16(lbl), connectivity_padding)
            # Apply penalty uniformly to non-boundary points
            psies[lbl][~dilated] += mask_penalty

        # Apply energy penalties for seed point misassignment
        for label, mask in seed_masks.items():

            coords = np.argwhere(mask)
            for (z, y, x) in coords:
                for other_label in range(1, num_labels + 1):
                    if other_label != label:
                        psies[other_label][z, y, x] += mask_penalty

        # auction algorithm
        epsilon0 = 10.0
        alpha = 5.0
        epsilonBar = 1e-6

        current_grid = calc.auction_assignment_3d(psies, target_volumes, grid_shape, num_labels, epsilon0, epsilonBar, alpha) # 71.6 sec (is ran num_iterations amount of times)

        # Calculate volumes
        # volumes = calc.calculate_3d_volumes(current_grid, num_labels)
        # total_vol = sum(volumes.values())
        # volume_conservation.append(total_vol == total_voxels)

        # region_history.append({
        #     'iteration': iter_num,
        #     'volumes': volumes
        # })

        print("\n")
        print(f"Euler characteristic: {calc.compute_surface_euler_characteristic(current_grid, background_label=1)}")
        print("\n")

        # Visualization
        # if iter_num % 1 == 0 or iter_num == num_iterations - 1:
            #visualize_3d_slices(current_grid, iter_num, num_labels)
            # visualize_3d_slices(current_grid, iter_num, num_labels + 1, view=(10,10),figsize=(15,15)) #boundary of grid()
            
            
            #np.save(f"pts4_1and4_1_second_{iter_num}.npy", next_grid) # save here
            #np.save(f"pts_granny_left_{iter_num}.npy", next_grid) # save here
            #np.save(f"pts_reidemeister{iter_num}.npy", next_grid) # save here
    
    print("3D Simulation complete")
    return next_grid



# start = time.perf_counter()
# end = time.perf_counter()
# print(f"time: {end - start} seconds")