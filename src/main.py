import knotgrowth.knotcurves as knots
import knotgrowth.gridgeneration as gg
import knotgrowth.visualizing as vis
import knotgrowth.linefield as lf
import knotgrowth.simulationloop as sl

import numpy as np

def main():
    print("started")

    # generate sigma matrix
    num_labels = 30
    grid_size = 70
    grid_shape = (grid_size, grid_size, grid_size)

    sigma = vis.generate_sigma_matrix(num_labels, 0, 2.9, 0.8, 4) # diag, background, adj, nonadj
    
    # Step 1, get points of the knot curve
    pts = knots.pts61_str2

    pts_upsampled = gg.upsample_points(pts, n_points_total=200)
    N = len(pts_upsampled)

    original_segment_lengths = np.zeros(N)
    for i in range(N):
        next_i = (i+1) % N
        original_segment_lengths[i] = np.linalg.norm(pts_upsampled[next_i] - pts_upsampled[i])

    pts_smooth = gg.relax_knot(pts_upsampled, original_segment_lengths)

    # Step 2, generate grid
    grid_side = 70    # Generates a grid of dimension 70^3
    num_cell_segments = 40  # The number of cell segments that partition the knot. Choose between (30,70). Efficiency is lower for larger
    scaling = 0.6    # The overall scaling of the initial knot. Choose between 0.55-0.8    
    tube_rad=2.4     #Initial tube radius. Choose around 2-3

    if not np.allclose(pts_smooth[0], pts_smooth[-1], atol=1e-3):
        pts_smooth = np.vstack([pts_smooth, pts_smooth[0]])

    label_grid = gg.points_to_voxel_grid(
        pts_smooth, 
        grid_size=grid_side,
        num_segments=num_cell_segments,
        scale=scaling,   # main was 0.65
        tube_radius=tube_rad,  #main 2
        padding=0.1
    )

    # To visualize with fixed parameters:
    #visualize_3d_slices(label_grid, 2, 60, view=(30, 30), figsize=(12, 12), cube_size=1.0)

    vis.plot_solid_voxels(label_grid, num_labels=100)


    num_iterations = 50 # Max number of iterations
    connectivity_padding = 3
    mask_penalty = np.inf #1e9
    num_labels = num_cell_segments + 1

    #seed points:
    penalty_radius = 0  #for seed regions
    depth, height, width = grid_shape
    total_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]

    region_history = []
    volume_conservation = [] 

    # Step 4
    sigma = vis.generate_sigma_matrix(num_labels, 0, 1.6, 0.8, 2.7)

    final_grid = sl.simulation_loop(label_grid, num_labels, grid_shape, penalty_radius, num_iterations, sigma, connectivity_padding, mask_penalty, region_history, volume_conservation)
    vis.print_volume_result(region_history, volume_conservation, total_voxels, num_labels)
    lined_grid = lf.draw_line_field(final_grid, grid_side, grid_shape, num_labels)
    
    vis.plot_solid_voxels(final_grid, num_labels=70)
    vis.plot_solid_voxels(lined_grid, 40)
    vis.plot_stereographic_projection(lined_grid)

    return



if __name__ == "__main__":
    main()
