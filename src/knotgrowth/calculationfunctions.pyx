import numpy as np
from scipy.ndimage import binary_dilation, label
import heapq
from numpy.fft import fftn, ifftn

def gaussian_kernel_3d(shape, dt):
    depth, height, width = shape
    z = np.arange(depth)
    y = np.arange(height)
    x = np.arange(width)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
    
    # Center frequencies around zero
    zz = np.where(zz > depth//2, depth - zz, zz)
    yy = np.where(yy > height//2, height - yy, yy)
    xx = np.where(xx > width//2, width - xx, xx)
    
    # Frequency steps
    freqStepX = 1.0 / width
    freqStepY = 1.0 / height
    freqStepZ = 1.0 / depth
    
    # Corrected: Use squared frequencies without squaring steps again
    freqStep = (xx**2) * freqStepX**2 + \
               (yy**2) * freqStepY**2 + \
               (zz**2) * freqStepZ**2
    
    return np.exp(-4 * np.pi**2 * dt * freqStep)

def psi_3d_optimized(grid, sigma_matrix, dt):
    depth, height, width = grid.shape
    grid_size = depth * height * width
    unique_labels = np.unique(grid)
    
    kernel_fft = gaussian_kernel_3d((depth, height, width), dt)
    
    chis = {}
    # chi_ffts = {}
    for label_val in unique_labels:
        chi = (grid == label_val).astype(float)
        chis[label_val] = chi
        # chi_ffts[label_val] = fftn(chi)
    
    phi_combined = np.zeros((len(unique_labels), depth, height, width))
    for i, target in enumerate(unique_labels):
        for j, source in enumerate(unique_labels):
            sigma_val = sigma_matrix[target-1, source-1]
            phi_combined[i] += sigma_val * chis[source]
    
    psies = {}
    for i, target in enumerate(unique_labels):
        ft_phi = fftn(phi_combined[i])
        ft_phi = ft_phi * kernel_fft / grid_size  # Correct normalization
        psi = np.real(ifftn(ft_phi))
        psies[target] = psi

    return psies


def dilate_boundary_3d(grid, lbl, iterations):
    mask = (grid == lbl)
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, :] = True
    struct[1, :, 1] = True
    struct[:, 1, 1] = True
    return binary_dilation(mask, structure=struct, iterations=iterations)

def enforce_connectivity_3d(grid, num_labels, min_size=5):
    connected = grid.copy()
    for lbl in range(1, num_labels + 1):
        mask = (grid == lbl)
        labeled, num_features = label(mask)
        
        if num_features > 0:
            component_ids, counts = np.unique(labeled, return_counts=True)
            component_ids = component_ids[1:]
            counts = counts[1:]
            
            if len(counts) > 0:
                largest_idx = component_ids[np.argmax(counts)]
                main_component = (labeled == largest_idx)
                
                for comp_id, count in zip(component_ids, counts):
                    if comp_id != largest_idx and count < min_size:
                        coords = np.argwhere(labeled == comp_id)
                        for coord in coords:
                            z, y, x = coord
                            neighbors = []
                            for dz, dy, dx in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                                nz, ny, nx = z+dz, y+dy, x+dx
                                if (0 <= nz < grid.shape[0] and 
                                    0 <= ny < grid.shape[1] and 
                                    0 <= nx < grid.shape[2]):
                                    neighbor_label = grid[nz, ny, nx]
                                    if neighbor_label != lbl and neighbor_label > 0:
                                        neighbors.append(neighbor_label)
                            
                            if neighbors:
                                new_label = max(set(neighbors), key=neighbors.count)
                                connected[z, y, x] = new_label
    return connected

def auction_assignment_3d(psies, target_volumes, grid_shape, num_labels, epsilon0, epsilonBar, alpha):
    depth, height, width = grid_shape
    nbNodes = depth * height * width
    nbCells = num_labels
    
    phi_dict = {}
    for lbl in range(1, num_labels + 1):
        phi_dict[lbl] = 1.0 - psies[lbl]
    
    phi = np.zeros((nbCells, nbNodes))
    for lbl in range(1, num_labels + 1):
        p = lbl - 1
        phi_grid = phi_dict[lbl]
        phi[p, :] = phi_grid.ravel()
    
    price = np.zeros(nbCells)
    isAssigned = np.full(nbNodes, -1)
    currentVolume = np.zeros(nbCells, dtype=int)
    bid = np.zeros(nbNodes)
    
    volumes = np.zeros(nbCells, dtype=int)
    for lbl in range(1, num_labels + 1):
        volumes[lbl - 1] = target_volumes[lbl]
        
    
    heaps = [[] for _ in range(nbCells)]
    epsilon = epsilon0
    
    while epsilon >= epsilonBar:
        nbAssignedNodes = 0
        isAssigned.fill(-1)
        currentVolume.fill(0)
        heaps = [[] for _ in range(nbCells)]
        
        l = 0
        while nbAssignedNodes < nbNodes:
            # Find next unassigned node
            while l < nbNodes and isAssigned[l] != -1:
                l += 1
            if l >= nbNodes:
                l = 0
                while l < nbNodes and isAssigned[l] != -1:
                    l += 1
                if l >= nbNodes:
                    break  # Critical fix: break outer loop
            
            pStar = 0
            pNext = 1
            valStar = phi[pStar, l] - price[pStar]
            valNext = phi[pNext, l] - price[pNext]
            
            if valStar < valNext:
                pStar, pNext = pNext, pStar
                valStar, valNext = valNext, valStar
            
            for p in range(2, nbCells):
                val = phi[p, l] - price[p]
                if val > valStar:
                    pNext = pStar
                    valNext = valStar
                    pStar = p
                    valStar = val
                elif val > valNext:
                    pNext = p
                    valNext = val
            
            bid_val = epsilon + phi[pStar, l] - phi[pNext, l] + price[pNext]
            bid[l] = bid_val
            
            if currentVolume[pStar] == volumes[pStar]:
                if heaps[pStar]:  # Critical fix: check heap not empty
                    _, evicted_node = heapq.heappop(heaps[pStar])
                    isAssigned[evicted_node] = -1
                    nbAssignedNodes -= 1
                    
                    heapq.heappush(heaps[pStar], (bid_val, l))
                    isAssigned[l] = pStar
                    
                    if heaps[pStar]:
                        price[pStar] = heaps[pStar][0][0]
            else:
                heapq.heappush(heaps[pStar], (bid_val, l))
                isAssigned[l] = pStar
                currentVolume[pStar] += 1
                nbAssignedNodes += 1
                
                if currentVolume[pStar] == volumes[pStar] and heaps[pStar]:
                    price[pStar] = heaps[pStar][0][0]
        
        epsilon = epsilon / alpha
    
    assignment = np.zeros(grid_shape, dtype=int)
    for idx in range(nbNodes):
        z = idx // (height * width)
        remainder = idx % (height * width)
        y = remainder // width
        x = remainder % width
        cell_idx = isAssigned[idx]
        if cell_idx >= 0:
            assignment[z, y, x] = cell_idx + 1
    
    return assignment

def boundary_of_grid(used_grid):
    # Create a copy of the grid
    
    grid_size = used_grid.shape[0]  # Assuming grid is cube-shaped

    # 1. Create a padded version of the grid (add 1-layer of background around edges)
    padded_grid = np.pad(used_grid, 1, mode='constant', constant_values=1)

    # 2. Vectorized neighbor checking
    # Create shifted versions of the grid for all 6 neighbors
    neighbor_checks = [
        padded_grid[2:, 1:-1, 1:-1],  # Down
        padded_grid[:-2, 1:-1, 1:-1],  # Up
        padded_grid[1:-1, 2:, 1:-1],   # Right
        padded_grid[1:-1, :-2, 1:-1],  # Left
        padded_grid[1:-1, 1:-1, 2:],   # Front
        padded_grid[1:-1, 1:-1, :-2]   # Back
    ]

    # 3. Identify boundary points using vectorized operations
    is_non_background = (used_grid != 1)
    has_background_neighbor = np.zeros_like(used_grid, dtype=bool)

    for neighbor in neighbor_checks:
        has_background_neighbor |= (neighbor == 1)

    boundary_mask = is_non_background & has_background_neighbor

    # 4. Create the boundary grid
    boundaried_grid = np.ones_like(used_grid)  # Start with all background
    boundaried_grid[boundary_mask] = used_grid[boundary_mask]  # Copy boundary values

    return boundaried_grid

def compute_surface_euler_characteristic(grid, background_label=1):
    grid_x, grid_y, grid_z = grid.shape
    
    # Handle empty case
    if np.all(grid == background_label):
        return 0

    # Initialize arrays for vertices and edges
    vertices_arr = np.zeros((grid_x+1, grid_y+1, grid_z+1), dtype=bool)
    edge_along_x = np.zeros((grid_x,   grid_y+1, grid_z+1), dtype=bool)
    edge_along_y = np.zeros((grid_x+1, grid_y,   grid_z+1), dtype=bool)
    edge_along_z = np.zeros((grid_x+1, grid_y+1, grid_z),   dtype=bool)
    face_count = 0

    # Get non-background voxel coordinates
    non_bg_indices = np.argwhere(grid != background_label)
    
    # Define 6 neighbor directions
    directions = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])

    for voxel in non_bg_indices:
        i, j, k = voxel
        for dx, dy, dz in directions:
            ni, nj, nk = i+dx, j+dy, k+dz
            
            # Check if neighbor is background
            if (ni < 0 or ni >= grid_x or 
                nj < 0 or nj >= grid_y or 
                nk < 0 or nk >= grid_z):
                is_bg = True
            else:
                is_bg = (grid[ni, nj, nk] == background_label)
            
            if not is_bg:
                continue
                
            face_count += 1  # Found boundary face
            
            # Get vertices for this face
            if (dx, dy, dz) == (1, 0, 0):
                vertices = [
                    (i+1, j,   k), (i+1, j,   k+1),
                    (i+1, j+1, k), (i+1, j+1, k+1)
                ]
            elif (dx, dy, dz) == (-1, 0, 0):
                vertices = [
                    (i, j,   k), (i, j,   k+1),
                    (i, j+1, k), (i, j+1, k+1)
                ]
            elif (dx, dy, dz) == (0, 1, 0):
                vertices = [
                    (i,   j+1, k), (i,   j+1, k+1),
                    (i+1, j+1, k), (i+1, j+1, k+1)
                ]
            elif (dx, dy, dz) == (0, -1, 0):
                vertices = [
                    (i,   j, k), (i,   j, k+1),
                    (i+1, j, k), (i+1, j, k+1)
                ]
            elif (dx, dy, dz) == (0, 0, 1):
                vertices = [
                    (i,   j,   k+1), (i,   j+1, k+1),
                    (i+1, j,   k+1), (i+1, j+1, k+1)
                ]
            else:  # (0, 0, -1)
                vertices = [
                    (i,   j,   k), (i,   j+1, k),
                    (i+1, j,   k), (i+1, j+1, k)
                ]
            
            # Mark vertices in boolean array
            for x, y, z in vertices:
                if 0 <= x <= grid_x and 0 <= y <= grid_y and 0 <= z <= grid_z:
                    vertices_arr[x, y, z] = True
            
            # Define edges (pairs of vertices)
            edges = [
                (vertices[0], vertices[1]),
                (vertices[0], vertices[2]),
                (vertices[1], vertices[3]),
                (vertices[2], vertices[3])
            ]
            
            # Mark edges in boolean arrays
            for va, vb in edges:
                if va[0] != vb[0]:  # Edge along x
                    x = min(va[0], vb[0])
                    y, z = va[1], va[2]
                    if 0 <= x < grid_x:
                        edge_along_x[x, y, z] = True
                elif va[1] != vb[1]:  # Edge along y
                    y = min(va[1], vb[1])
                    x, z = va[0], va[2]
                    if 0 <= y < grid_y:
                        edge_along_y[x, y, z] = True
                elif va[2] != vb[2]:  # Edge along z
                    z = min(va[2], vb[2])
                    x, y = va[0], va[1]
                    if 0 <= z < grid_z:
                        edge_along_z[x, y, z] = True
    
    # Count elements
    V = vertices_arr.sum()
    E = edge_along_x.sum() + edge_along_y.sum() + edge_along_z.sum()
    F = face_count
    return V - E + F

def calculate_3d_volumes(grid, num_labels):
    volumes = {}
    for lbl in range(1, num_labels + 1):
        volumes[lbl] = np.sum(grid == lbl)
    return volumes