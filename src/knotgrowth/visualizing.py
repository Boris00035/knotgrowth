import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import plotly.graph_objects as go
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


def get_40_cmap():
    # Use a well-spaced set of colors from "hsv"
    base = plt.cm.get_cmap("hsv", 40)  # 40 evenly spaced hues
    colors = [base(i) for i in range(40)]        
    random.shuffle(colors)
    return mcolors.ListedColormap(colors, name="40colors")

cmap = get_40_cmap()


def visualize_3d_slices(grid, iteration, num_labels, view=(30, 30), figsize=(12,12), cube_size=1.0):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    # Use tab20 colormap
    #cmap = plt.get_cmap("tab20", num_labels)
    
    

    # Legend elements for labels
    legend_elements = []

    for label in range(2, num_labels + 1):  # Skip background (label 1)

        coords = np.column_stack(np.nonzero(grid == label))

        if len(coords) == 0:
            continue

        color = cmap((label - 2) % 20)

        # Shift coordinates to center the cubes at (x, y, z)
        x = coords[:, 0] - cube_size / 2
        y = coords[:, 1] - cube_size / 2
        z = coords[:, 2] - cube_size / 2

        ax.bar3d(
            x, y, z,
            dx=cube_size, dy=cube_size, dz=cube_size,
            color=color, alpha=0.9, shade=True
        )

        legend_elements.append(Patch(facecolor=color, label=f"Label {label}"))

    ax.set_xlim(0, grid.shape[0])
    ax.set_ylim(0, grid.shape[1])
    ax.set_zlim(0, grid.shape[2])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    ax.view_init(elev=view[0], azim=view[1])
    plt.pause(0.1)
    plt.close(fig)


# Example usage:  plot_solid_voxels(grid_smooth, num_labels=21)


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.dot(a, b)
    if np.allclose(c, 1.0):
        return np.eye(3)
    if np.allclose(c, -1.0):
        axis = np.cross(a, np.array([1,0,0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0,1,0]))
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2*np.outer(axis, axis)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)
    return np.eye(3) + K + K @ K * ((1 - c)/(s**2))

def stereographic_projection_yz(points, pole):
    if pole == "north":
        denom = 1 - points[:,0]
    else:  # pole == "south"
        denom = 1 + points[:,0]

    y = points[:,1] / denom
    z = points[:,2] / denom

    return np.vstack([y, z]).T

def stereographic_projection_xy(points, pole):
    if pole == "north":
        denom = 1 - points[:,2]
    else:  # pole == "south"
        denom = 1 + points[:,2]
    x = points[:,0] / denom
    y = points[:,1] / denom
    return np.vstack([x,y]).T

# def split_and_project(label_grid, center, n):
#     # voxel coords for label=2
#     coords = np.argwhere(label_grid == 2).astype(float)
#     rel = coords - center
#     unit = rel / np.linalg.norm(rel, axis=1)[:,None]

#     # rotate n → z-axis
#     R = rotation_matrix_from_vectors(n, np.array([0,0,1]))
#     rot = unit @ R.T

#     # split hemispheres
#     upper = rot[rot[:,2] >= 0]   # z ≥ 0
#     lower = rot[rot[:,2] < 0]

#     upper_proj = stereographic_projection(upper, pole="south")
#     lower_proj = stereographic_projection(lower, pole="north")

#     return upper_proj, lower_proj


# def plot_stereographic_projection(lined_grid):
#     N = 70

#     griddd = np.copy(lined_grid)
#     center = np.array([N/2, N/2, N/2])

#     n = np.array([1, -0.5, 0])

#     upper_proj, lower_proj = split_and_project(griddd, center, n)

#     # Plot side by side
#     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
#     ax1.scatter(upper_proj[:,0], upper_proj[:,1], s=6, c="C0")
#     ax1.set_title("Upper hemisphere")
#     ax1.set_aspect("equal")

#     ax2.scatter(lower_proj[:,0], lower_proj[:,1], s=6, c="C1")
#     ax2.set_title("Lower hemisphere")
#     ax2.set_aspect("equal")

#     plt.show()

def print_volume_result(region_history, volume_conservation, total_voxels, num_labels):
    print("\nVolume conservation report:")
    #np.save(f"unknotmediaa_{dtdt}.npy", next_grid) # save here

    for i, conserved in enumerate(volume_conservation):
        if not conserved:
            total_vol = sum(region_history[i]['volumes'].values())
            print(f"Iteration {i}: Volume mismatch! {total_vol} vs {total_voxels}")
        else:
            print(f"Iteration {i}: Volume conserved")

    print("\nFinal volumes:")
    for lbl in range(1, num_labels + 1):
        vol = region_history[-1]['volumes'].get(lbl, 0)
        print(f"Label {lbl}: {vol} voxels")

    plt.figure(figsize=(10, 6))
    for lbl in range(1, num_labels + 1):
        vols = [h['volumes'].get(lbl, 0) for h in region_history]
        plt.plot(vols, label=f"Label {lbl}")

    plt.title("Volume Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()