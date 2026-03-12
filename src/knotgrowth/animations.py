from tqdm.auto import trange
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from enum import Enum
import numpy as np
import os
from datetime import datetime
import PIL
import imageio

import knotgrowth.visualizing as vis
import knotgrowth.growth as gr


class possible_inputs(Enum):
    unknot_circle = "unknot/circle/"
    unknot_dent = "unknot/dent/"
    unknot_twist = "unknot/twist/"
    trefoil_dent = "trefoil/dent/"
    trefoil_twist = "trefoil/twist/"

# maybe create a dict like "simulation parameters"
def create_boundary_animation(grid_size, NOI, NOF, num_cell_segments, input, animation_duration=0):

    animation_input = "animations/" + input.value

    assert len(os.listdir(animation_input)) == NOF, "the amount of frames does not match the amount of frame data from the animation, should probably reexport the animation from blender"

    print(f"parameters: NOI: {NOI}, grid_size: {grid_size}")
    
    fig = go.Figure(data = [go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,  # Size of the squares
                opacity=1.0,  # Fully opaque
                symbol='square',  # Square markers
                line=dict(width=0),  # No border line
            ),
            name="boundary"
        )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, grid_size], autorange=False),
            yaxis=dict(range=[0, grid_size], autorange=False),
            zaxis=dict(range=[0, grid_size], autorange=False),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        width=1080,
        height=1080,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])],
        )],
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[str(k)], dict(mode="immediate", frame=dict(duration=animation_duration, redraw=True))],
                    "label": str(k)
                }
                for k in range(1, NOF + 1)
            ]
        }]
    )

    animation_frames = []
    output_folder = "output/" + "raw/" + animation_input + datetime.today().strftime('%Y-%m-%d') + "/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame in trange(1, NOF + 1, desc='frame loop'):
        points = np.load(animation_input + f"frame{frame}" + ".npy")
        lined_grid = gr.get_boundary_after_growth(points, NOI, num_cell_segments, grid_size)
        
        mask = (lined_grid == 2)
    
        boundary_points = np.where(mask)
        
        frame_data = dict(
            type="scatter3d",
            x=boundary_points[0],
            y=boundary_points[1],
            z=boundary_points[2],
        )

        np.save(output_folder + f"frame{frame}" + ".npy", boundary_points)
        animation_frames.append(go.Frame(data = [frame_data], name=str(len(animation_frames))))

    fig.frames = animation_frames

    fig.show()

def create_boundary_animation_3d_from_files(grid_size, input, animation_duration=0):
    # Minus 1 for the parameters.txt file

    output_data_location = "output/" + input.value

    NOF = len(os.listdir(output_data_location)) - 1

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "image"}]],
        column_widths=[0.7, 0.3]
    )


    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,
                opacity=1.0,
                symbol='square',
                line=dict(width=0),
            ),
            name="boundary"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Image(z=np.zeros((10,10))),
        row=1,
        col=2
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, grid_size], autorange=False),
            yaxis=dict(range=[0, grid_size], autorange=False),
            zaxis=dict(range=[0, grid_size], autorange=False),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        width=2200,
        height=1000,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])],
        )],
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[str(k)], dict(mode="immediate", frame=dict(duration=animation_duration, redraw=True))],
                    "label": str(k)
                }
                for k in range(0, NOF)
            ]
        }]
    )

    animation_frames = []

    for frame in range(1, NOF + 1):
        points = np.load(output_data_location + f"frame{frame}" + ".npy")

        frame_data = dict(
            type="scatter3d",
            x=points[0],
            y=points[1],
            z=points[2],
        )

        img_path = "animations/" + input.value + "animation/" + f"{frame:04d}.png"
        img = PIL.Image.open(img_path)
        image_data = go.Image(z=img)


        animation_frames.append(go.Frame(data = [frame_data, image_data], name=str(len(animation_frames))))

    fig.frames = animation_frames

    fig.show()

def create_boundary_animation_sp_from_files(grid_size, input, animation_duration=0, save_animation=False):
    # Minus 1 for the parameters.txt file

    output_data_location = "output/" + input.value

    NOF = len(os.listdir(output_data_location)) - 1

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "image"}]],
        column_widths=[0.7, 0.3]
    )


    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(
                size=8,
                opacity=1.0,
                symbol='square',
                line=dict(width=0),
            ),
            name="boundary"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Image(z=np.zeros((10,10))),
        row=1,
        col=2
    )
    
    fig.update_layout(
        xaxis1=dict(range=[-3, 3]),
        yaxis1=dict(range=[-3, 3], scaleanchor="x"),
        width=1400,
        height=1000,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])],
        )],
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[str(k)], dict(mode="immediate", frame=dict(duration=animation_duration, redraw=True))],
                    "label": str(k)
                }
                for k in range(0, NOF)
            ]
        }]
    )

    animation_frames = []

    for frame in range(1, NOF + 1):
        offset = np.array([
            round(grid_size / 2),
            round(grid_size / 2),
            round(grid_size / 2)
        ])

        points = np.load(output_data_location + f"frame{frame}" + ".npy").T - offset
        unit_points = points / np.linalg.norm(points, axis=1)[:, None]
        flat_points = vis.stereographic_projection_yz(unit_points, "north")

        # print(flat_points)

        frame_data = dict(
            type="scatter",
            x=flat_points.T[0],
            y=flat_points.T[1],
        )

        img_path = "animations/" + input.value + "animation/" + f"{frame:04d}.png"
        img = PIL.Image.open(img_path)
        image_data = go.Image(z=img)


        animation_frames.append(go.Frame(data = [frame_data, image_data], name=str(len(animation_frames))))

    fig.frames = animation_frames
    fig.show()

    if save_animation:
        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            img_bytes = fig.to_image(format="jpg")
            images.append(imageio.imread(img_bytes))

        imageio.mimsave("output/" + f"{datetime.today().strftime('%Y-%m-%d')}.mp4", images, fps=1)


