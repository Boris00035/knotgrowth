import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy.spatial import KDTree

def upsample_points(pts, n_points_total=200):
    N = len(pts)
    
    # Calculate segment lengths
    segment_lengths = np.zeros(N)
    for i in range(N):
        next_i = (i+1) % N
        segment_lengths[i] = np.linalg.norm(pts[next_i] - pts[i])
    
    total_length = np.sum(segment_lengths)
    points_per_segment = [max(3, int(n_points_total * length / total_length)) for length in segment_lengths]
    
    # Adjust total points
    current_total = sum(points_per_segment)
    i = 0
    while current_total != n_points_total:
        if current_total < n_points_total:
            points_per_segment[i % N] += 1
            current_total += 1
        elif current_total > n_points_total and points_per_segment[i % N] > 3:
            points_per_segment[i % N] -= 1
            current_total -= 1
        i += 1

    pts_new = []
    for i in range(N):
        start = pts[i]
        end = pts[(i+1) % N]
        segment_pts = np.linspace(start, end, points_per_segment[i], endpoint=False)
        pts_new.extend(segment_pts)
    
    return np.array(pts_new, dtype=float)


def compute_forces(pts, original_lengths, k_spring=300, k_rep=0.05, k_curvature=100):
    N = len(pts)
    F = np.zeros_like(pts)
    
    
    for i in range(N):
        next_i = (i+1) % N
        vec = pts[next_i] - pts[i]
        curr_len = np.linalg.norm(vec)
        if curr_len > 1e-6:
            
            force_mag = k_spring * (curr_len - original_lengths[i]) / curr_len
            force_vec = force_mag * vec
            F[i] += force_vec
            F[next_i] -= force_vec
    
    # prevent sharp corners
    for i in range(N):
        prev_i = (i-1) % N
        next_i = (i+1) % N
        
        
        vec_to_prev = pts[prev_i] - pts[i]
        vec_to_next = pts[next_i] - pts[i]
        
        
        curvature_vec = (vec_to_prev + vec_to_next) / 2
        
        
        smoothing_force = k_curvature * curvature_vec
        F[i] += smoothing_force
    
    
    dist_matrix = cdist(pts, pts)
    np.fill_diagonal(dist_matrix, np.inf)
    
    chain_indices = np.arange(N)
    index_diff = np.abs(chain_indices[:, None] - chain_indices[None, :])
    chain_distance = np.minimum(index_diff, N - index_diff) 
    
    
    repulsion_threshold = np.mean(original_lengths) * 1.5
    
    for i in range(N):
        for j in range(i+1, N):

            if chain_distance[i, j] <= 5:  
                continue
                
            # Calculate repulsion if points are close in space
            dist = dist_matrix[i, j]
            if dist < repulsion_threshold: 
                direction = pts[j] - pts[i]
                unit_dir = direction / dist
                
                # Soft repulsion that increases as distance decreases
                repulsion_strength = k_rep * (1 - dist/repulsion_threshold)**2 ##
                repulsion = repulsion_strength * unit_dir / dist
                F[i] -= repulsion
                F[j] += repulsion
    
    return F



def relax_knot(pts, original_lengths, max_steps=4000, initial_dt=0.001, min_dt=0.0001):
    pts = pts.copy()
    current_dt = initial_dt
    prev_energy = float('inf')
    
    avg_seg_len = np.mean(original_lengths)
    
    for step in range(max_steps):
        F = compute_forces(pts, original_lengths)
        force_mags = np.linalg.norm(F, axis=1)
        max_force = np.max(force_mags)
        energy = np.sum(force_mags**2)
        
        if max_force > 500:
            current_dt = max(min_dt, current_dt * 0.8)
        elif max_force > 100:
            current_dt = max(min_dt, current_dt * 0.9)
        else:
            current_dt = min(initial_dt, current_dt * 1.02)
        
        displacement = current_dt * F
        max_disp = np.max(np.linalg.norm(displacement, axis=1))
        if max_disp > avg_seg_len * 0.1:  
            displacement *= (avg_seg_len * 0.1) / max_disp
        
        pts += displacement
        
        # Check for convergence
        if step % 100 == 0:
            energy_change = abs(prev_energy - energy)
            if energy_change < 1e-5 * energy:
                break
            prev_energy = energy
    
    return pts

def points_to_grid(pts, grid_size=60, padding=0.1):

    min_vals = np.min(pts, axis=0)
    max_vals = np.max(pts, axis=0)
    center = (min_vals + max_vals) / 2
    range_vals = max_vals - min_vals
    max_range = np.max(range_vals)
    
    scale_factor = (1 - 2*padding) / max_range
    
    
    pts_centered = (pts - center) * scale_factor + 0.5
    
    pts_scaled = (pts_centered * (grid_size - 1)).astype(int)
    
    grid = np.ones((grid_size, grid_size, grid_size), dtype=int)
    for x, y, z in pts_scaled:
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            grid[x, y, z] = 2
    
    return grid

def points_to_voxel_grid(pts, grid_size, num_segments, scale, tube_radius, padding=0.1):
    if not np.allclose(pts[0], pts[-1], atol=1e-6):
        pts_closed = np.vstack([pts, pts[0]])
    else:
        pts_closed = pts

    diffs = pts_closed[1:] - pts_closed[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    
    valid_indices = np.where(dists > 1e-8)[0]
    if len(valid_indices) == 0:
        raise ValueError("All points are identical")
    
    pts_closed = np.vstack([
        pts_closed[0],
        pts_closed[1:][valid_indices]
    ])
    
    diffs = pts_closed[1:] - pts_closed[:-1]
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.cumsum(dists)
    total_length = cum_dists[-1]
    s_points = np.concatenate([[0], cum_dists])
    
    n_dense = 1000
    s_interp = np.linspace(0, total_length, n_dense, endpoint=False)
    
    interp_x = interp1d(s_points, pts_closed[:, 0], kind='linear', assume_sorted=True)
    interp_y = interp1d(s_points, pts_closed[:, 1], kind='linear', assume_sorted=True)
    interp_z = interp1d(s_points, pts_closed[:, 2], kind='linear', assume_sorted=True)
    
    pts_dense = np.column_stack([
        interp_x(s_interp),
        interp_y(s_interp),
        interp_z(s_interp)
    ])
    
    segment_ids = (np.arange(n_dense) * num_segments // n_dense) % num_segments
    
    min_vals = np.min(pts_dense, axis=0)
    max_vals = np.max(pts_dense, axis=0)
    center = (min_vals + max_vals) / 2
    range_vals = max_vals - min_vals
    max_range = np.max(range_vals)
    
    scale_factor = (1 - 2 * padding) / max_range
    scale_factor = 10
    pts_dense = (pts_dense - center) * scale_factor * scale
    pts_dense += grid_size / 2  # Center in grid
    
    tree = KDTree(pts_dense)
    
    # Prepare label grid (background = 1)
    label_grid = np.ones((grid_size, grid_size, grid_size), dtype=int)
    
    # Create coordinate grid
    z, y, x = np.indices((grid_size, grid_size, grid_size))
    coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    
    batch_size = 10000
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        dists, idxs = tree.query(batch, k=1, distance_upper_bound=tube_radius+1e-6)
        
        # Find points within tube radius
        valid_mask = dists <= tube_radius
        
        # Get valid indices and labels
        valid_idxs = idxs[valid_mask]
        valid_labels = segment_ids[valid_idxs] + 2  # +2 to start labels from 2
        valid_coords = batch[valid_mask]
        
        # Assign to grid
        x_coords = valid_coords[:, 0].astype(int)
        y_coords = valid_coords[:, 1].astype(int)
        z_coords = valid_coords[:, 2].astype(int)
        
        valid_indices = (x_coords >= 0) & (x_coords < grid_size) & \
                        (y_coords >= 0) & (y_coords < grid_size) & \
                        (z_coords >= 0) & (z_coords < grid_size)
        
        x_valid = x_coords[valid_indices]
        y_valid = y_coords[valid_indices]
        z_valid = z_coords[valid_indices]
        labels_valid = valid_labels[valid_indices]
        
        label_grid[z_valid, y_valid, x_valid] = labels_valid
    
    return label_grid