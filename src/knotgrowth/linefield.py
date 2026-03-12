import numpy as np
from scipy.ndimage import zoom
import knotgrowth.calculationfunctions as calc

def draw_line_field(next_grid, grid_size, num_labels):
    our_grid  = np.copy(next_grid)
    our_grid = calc.boundary_of_grid(our_grid)

    grid_shape = [grid_size, grid_size, grid_size]

    # Parameters
    upsample_factor = 5
    sphere_center = (np.round(grid_size / 2), np.round(grid_size / 2), np.round(grid_size / 2))

    # Example: our_grid is your input voxel grid
    # Upscale the grid (nearest neighbor so labels are preserved)
    high_res_grid = zoom(our_grid, upsample_factor, order=0)  

    high_res_size = high_res_grid.shape[0]
    high_res_center = tuple(c * upsample_factor for c in sphere_center)

    # Create high-res sphere grid
    high_res_sphere_grid = np.ones_like(high_res_grid)

    # Spherical projection at high resolution
    for i in range(high_res_size):
        for j in range(high_res_size):
            for k in range(high_res_size):

                if high_res_grid[i, j, k] != 1:
                    x_vec = i - high_res_center[0]
                    y_vec = j - high_res_center[1]
                    z_vec = k - high_res_center[2]

                    norm = np.linalg.norm([x_vec, y_vec, z_vec])
                    if norm == 0:
                        continue

                    # Project to radius (e.g., 0.9 * max radius to avoid overflow)
                    radius = 0.9 * high_res_size / 2
                    nx = int(high_res_center[0] + (x_vec / norm) * radius)
                    ny = int(high_res_center[1] + (y_vec / norm) * radius)
                    nz = int(high_res_center[2] + (z_vec / norm) * radius)

                    high_res_sphere_grid[nx, ny, nz] = high_res_grid[i, j, k]

                    
    final_sphere_grid = np.ones((grid_size, grid_size, grid_size), dtype=high_res_grid.dtype)

    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                block = high_res_sphere_grid[
                    x*upsample_factor:(x+1)*upsample_factor,
                    y*upsample_factor:(y+1)*upsample_factor,
                    z*upsample_factor:(z+1)*upsample_factor
                ]
                # Assign the most frequent non-background label (or 1 if empty)
                labels, counts = np.unique(block, return_counts=True)
                if len(labels) > 1:
                    labels = labels[labels != 1]  # remove background
                final_sphere_grid[x, y, z] = labels[0] if len(labels) > 0 else 1

    my_grid = np.copy(final_sphere_grid).astype(int)

    lined_grid = np.ones_like(my_grid)

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):

                my_label = my_grid[i, j, k]

                if my_label == 1:
                    continue

                ngbh = six_neighbors(grid_shape, (i, j, k))
                index_set = set()
                for l in ngbh:
                    index_set.add(my_grid[l[0], l[1], l[2]])

                if any(v not in (my_label, 1) for v in index_set) and (not my_label + 1 in index_set) and (not my_label - 1 in index_set):

                    if my_label == 2 and (num_labels in index_set):
                        continue
                    elif my_label == num_labels and (2 in index_set):
                        continue

                    lined_grid[i, j, k] = 2
    
    return lined_grid


def six_neighbors(grid_shape, coord):

    x, y, z = coord
    neighbors = [
        (x-1, y, z),
        (x+1, y, z),
        (x, y-1, z),
        (x, y+1, z),
        (x, y, z-1),
        (x, y, z+1)
    ]

    valid_neighbors = [
        (i, j, k)
        for (i, j, k) in neighbors
        if 0 <= i < grid_shape[0]
        and 0 <= j < grid_shape[1]
        and 0 <= k < grid_shape[2]
    ]

    return valid_neighbors