import numpy as np

import knotgrowth.gridgeneration as gg
import knotgrowth.simulationloop as sl
import knotgrowth.calculationfunctions as calc


def get_grid_after_growth(starting_points, animation_input, num_iterations, num_labels, grid_size, frame_num, knot_relaxation=False, save_growth_process=False):

    # generate sigma matrix
    scaling = 0.6    # The overall scaling of the initial knot. Choose between 0.55-0.8    
    tube_rad=2.4     #Initial tube radius. Choose around 2-3
    connectivity_padding = 3
    mask_penalty = np.inf #1e9

    sigma = calc.generate_sigma_matrix(num_labels, 0, 2.9, 0.8, 4) # diag, background, adj, nonadj

    # Step 1, get points of the knot curve

    N = len(starting_points)

    original_segment_lengths = np.zeros(N)
    for i in range(N):
        next_i = (i+1) % N
        original_segment_lengths[i] = np.linalg.norm(starting_points[next_i] - starting_points[i])


    # This makes the knot into a good shape, not needed if the shape is already as we want
    if knot_relaxation:
        pts_smooth = gg.relax_knot(starting_points, original_segment_lengths) # This function takes 27 sec
    else:
        pts_smooth = starting_points


    # Step 2, generate grid
    if not np.allclose(pts_smooth[0], pts_smooth[-1], atol=1e-3):
        pts_smooth = np.vstack([pts_smooth, pts_smooth[0]])

    label_grid = gg.points_to_voxel_grid(
        pts_smooth, 
        grid_size=grid_size,
        num_segments=num_labels - 1,
        scale=scaling,   # main was 0.65
        tube_radius=tube_rad,  #main 2
        padding=0.1
    )

    #seed points:
    penalty_radius = 0  #for seed regions
    region_history = []
    volume_conservation = [] 

    # Step 4
    sigma = calc.generate_sigma_matrix(num_labels, 0, 1.6, 0.8, 2.7)

    final_grid, boundary = sl.simulation_loop(label_grid, num_labels, grid_size, penalty_radius, num_iterations, sigma, connectivity_padding, mask_penalty, region_history, volume_conservation, frame_num, animation_input, save_growth_process=save_growth_process)

    return final_grid, boundary