from tqdm.auto import trange
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from enum import Enum
import numpy as np
import os
from datetime import datetime
import PIL.Image as im
import imageio
import matplotlib.pyplot as plt

import knotgrowth.visualizing as vis
import knotgrowth.growth as gr


class possible_inputs(Enum):
    unknot_circle = "unknot/circle/"
    unknot_dent = "unknot/dent/"
    unknot_twist = "unknot/twist/"
    unknot_double_twist = "unknot/double_twist/"
    trefoil_trefoil = "trefoil/trefoil/"
    test = "test/"
    # trefoil_dent = "trefoil/dent/"
    # trefoil_twist = "trefoil/twist/"


def generate_grids_after_growth(grid_size, NOI, NOF, num_labels, input, start_frame=1, save_grid=False, save_boundary=False, save_growth_process=False):

    animation_input = "animations/" + input.value
    # + 1 because of the animation folder (the blender rendered animation of the changing knot)
    assert len(os.listdir(animation_input)) == NOF + 1, "the amount of frames does not match the amount of frame data from the animation, should probably reexport the animation from blender"
    
    output_folder = "output/" + "raw/" + datetime.today().strftime('%Y-%m-%d_%H-%M-%S') + animation_input
    output_folder_grid = output_folder + "/grid/"
    output_folder_boundary = output_folder + "/boundary/"
    
    if not os.path.exists(output_folder_grid):
        os.makedirs(output_folder_grid)
    if not os.path.exists(output_folder_boundary):
        os.makedirs(output_folder_boundary)

    for frame_num in trange(start_frame, NOF + 1, desc='frame loop'):
        points = np.load(animation_input + f"frame{frame_num}" + ".npy")
        grid, boundary = gr.get_grid_after_growth(points, NOI, num_labels, frame_num, grid_size, save_growth_process)

        if save_grid:
            np.save(output_folder_grid + f"frame{frame_num}" + ".npy", grid)
        if save_boundary:
            np.save(output_folder_boundary + f"frame{frame_num}" + ".npy", boundary)

def view_grid_animation_3d(input, num_labels,  animation_duration=0, show_animation=True, save_video=False, save_html=False):
    grid_output_data_location = "output/" + input.value + "grid/"
    boundary_output_data_location = "output/" + input.value + "boundary/"
    
    assert len(os.listdir(grid_output_data_location)) == len(os.listdir(boundary_output_data_location)), f"grid frames: {len(os.listdir(grid_output_data_location))}, boundary frames: {len(os.listdir(boundary_output_data_location))}"
    NOF = len(os.listdir(grid_output_data_location))
    
    
    # Get tab20 colors
    tab20_colors = plt.get_cmap("tab20").colors
    colors = ['rgb(%d,%d,%d)' % (r*255, g*255, b*255) for (r, g, b) in tab20_colors]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "image"}]],
        column_widths=[0.7, 0.3]
    )

    # add trace for each colour
    for label in range(2, num_labels + 1):        
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,  # Size of the squares
                color=colors[(label-2-1) % 20],  # Solid color for this label
                opacity=1.0,  # Fully opaque
                symbol='square',  # Square markers
                line=dict(width=0),  # No border line
            ),
            name=f'Label{label}'
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    # add trace for boundary
    fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,  # Size of the squares
                color=colors[(label-2-1) % 20],  # Solid color for this label
                opacity=1.0,  # Fully opaque
                symbol='square',  # Square markers
                line=dict(width=0),  # No border line
            ),
            name='boundary'
        ),
        row=1,
        col=1
    )

    # add trace for the blender pictures
    fig.add_trace(
        go.Image(
            z=np.zeros((10,10)),
            name="animation"
            ),
        row=1,
        col=2,
    )
    
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # Add play button
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


    for frame_num in range(1, NOF + 1):
        grid = np.load(grid_output_data_location + f"frame{frame_num}" + ".npy")
        traces = []

        for label in range(2, num_labels + 1):
            mask = (grid == label)
            if np.any(mask):
                x, y, z = np.where(mask)
            else:    
                x, y, z = [], [], []

            # Get coordinates
            
            trace = dict(
                type="scatter3d",
                mode='markers',
                x=x,
                y=y,
                z=z,
                marker=dict(
                    size=8,
                    color=colors[(label-2-1) % 20],
                    symbol="square",
                    opacity=1.0
                ),
                name=f"Label{label}",
            )

            traces.append(trace)

        # update boundary animation
        boundary_points = np.load(boundary_output_data_location + f"frame{frame_num}" + ".npy")
        boundary_trace = dict(
            type="scatter3d",
            mode='markers',
            x=boundary_points[0],
            y=boundary_points[1],
            z=boundary_points[2],
            marker=dict(
                size=8,
                color="Black",
                symbol="square",
                opacity=1.0
            ),
            name="boundary",
        )
        traces.append(boundary_trace)

        # update blender animation
        img_path = "animations/" + input.value + "animation/" + f"{frame_num:04d}.png"
        img = im.open(img_path)
        traces.append(dict(
            type="image",
            z=img,
            ))
                
        new_frame = go.Frame(
            data=traces,
            name=f"{frame_num}",
        )
        
        animation_frames.append(new_frame)

    fig.frames = animation_frames

    initial_frame = 0
    fig.update(data=fig.frames[initial_frame].data)

    if show_animation:
        fig.show()

    if save_video:
        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            img_bytes = fig.to_image(format="jpg")
            images.append(imageio.imread(img_bytes))

        imageio.mimsave("output/videos/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.mp4", images, fps=1)
    
    if save_html:
        fig.write_html("output/interactive_html/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.html", auto_play=False)


def view_growth_process(input, num_labels, frame_num=None,  animation_duration=0, show_animation=True, save_video=False, save_html=False):
    output_folder = "output/" + input + "growth_process/" + f"frame{frame_num}/"
    grid_output_data_location = output_folder + "grid/"
    boundary_output_data_location = output_folder + "boundary/"

    num_grid_frames = len(os.listdir(grid_output_data_location))
    num_boundary_frames = len(os.listdir(grid_output_data_location))
    NOF = num_grid_frames

    assert num_grid_frames == num_boundary_frames, f"grid frames: {num_grid_frames}, boundary frames: {num_boundary_frames}"
    
    if frame_num is None:
        frame_num = NOF

    
    # Get tab20 colors
    tab20_colors = plt.get_cmap("tab20").colors
    colors = ['rgb(%d,%d,%d)' % (r*255, g*255, b*255) for (r, g, b) in tab20_colors]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "image"}]],
        column_widths=[0.7, 0.3]
    )

    # add trace for each colour
    for label in range(2, num_labels + 1):        
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,  # Size of the squares
                color=colors[(label-2-1) % 20],  # Solid color for this label
                opacity=1.0,  # Fully opaque
                symbol='square',  # Square markers
                line=dict(width=0),  # No border line
            ),
            name=f'Label{label}'
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )

    # add trace for boundary
    fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=8,  # Size of the squares
                color=colors[(label-2-1) % 20],  # Solid color for this label
                opacity=1.0,  # Fully opaque
                symbol='square',  # Square markers
                line=dict(width=0),  # No border line
            ),
            name='boundary'
        ),
        row=1,
        col=1
    )

    # add trace for the blender pictures
    fig.add_trace(
        go.Image(
            z=np.zeros((10,10)),
            name="animation"
            ),
        row=1,
        col=2,
    )
    
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
            )
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # Add play button
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


    for iter_num in range(1, NOF + 1):
        grid = np.load(grid_output_data_location + f"iter{iter_num}" + ".npy")
        traces = []

        for label in range(2, num_labels + 1):
            mask = (grid == label)
            if np.any(mask):
                x, y, z = np.where(mask)
            else:    
                x, y, z = [], [], []

            # Get coordinates
            
            trace = dict(
                type="scatter3d",
                mode='markers',
                x=x,
                y=y,
                z=z,
                marker=dict(
                    size=8,
                    color=colors[(label-2-1) % 20],
                    symbol="square",
                    opacity=1.0
                ),
                name=f"Label{label}",
            )

            traces.append(trace)

        # update boundary animation
        boundary_points = np.load(boundary_output_data_location + f"iter{iter_num}" + ".npy")
        boundary_trace = dict(
            type="scatter3d",
            mode='markers',
            x=boundary_points[0],
            y=boundary_points[1],
            z=boundary_points[2],
            marker=dict(
                size=8,
                color="Black",
                symbol="square",
                opacity=1.0
            ),
            name="boundary",
        )
        traces.append(boundary_trace)

        # update blender animation
        img_path = "animations/" + input.value + "animation/" + f"{iter_num:04d}.png"
        img = im.open(img_path)
        traces.append(dict(
            type="image",
            z=img,
            ))
                
        new_frame = go.Frame(
            data=traces,
            name=f"{iter_num}",
        )
        
        animation_frames.append(new_frame)

    fig.frames = animation_frames

    initial_frame = 0
    fig.update(data=fig.frames[initial_frame].data)

    if show_animation:
        fig.show()

    if save_video:
        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            img_bytes = fig.to_image(format="jpg")
            images.append(imageio.imread(img_bytes))

        imageio.mimsave("output/videos/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.mp4", images, fps=1)
    
    if save_html:
        fig.write_html("output/interactive_html/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.html", auto_play=False)



def view_boundary_animation_sp(input, grid_size, animation_duration=0, save_video=False, save_html=False):

    output_data_location = "output/" + input.value + "boundary/"
    NOF = len(os.listdir(output_data_location))

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
        img = im.open(img_path)
        image_data = go.Image(z=img)


        animation_frames.append(go.Frame(data = [frame_data, image_data], name=str(len(animation_frames))))

    fig.frames = animation_frames

    initial_frame = 0
    fig.update(data=fig.frames[initial_frame].data)

    fig.show()

    if save_video:
        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            img_bytes = fig.to_image(format="jpg")
            images.append(imageio.imread(img_bytes))

        imageio.mimsave("output/videos/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.mp4", images, fps=1)
    
    if save_html:
        fig.write_html("output/interactive_html/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.html", auto_play=False)



# def view_boundary_animation_3d(input, animation_duration=0, save_video=False, save_html=False):

#     output_data_location = "output/" + input.value + "boundary/"
#     NOF = len(os.listdir(output_data_location))

#     fig = make_subplots(
#         rows=1,
#         cols=2,
#         specs=[[{"type": "scene"}, {"type": "image"}]],
#         column_widths=[0.7, 0.3]
#     )


#     fig.add_trace(
#         go.Scatter3d(
#             x=[0],
#             y=[0],
#             z=[0],
#             mode='markers',
#             marker=dict(
#                 size=8,
#                 opacity=1.0,
#                 symbol='square',
#                 line=dict(width=0),
#             ),
#             name="boundary"
#         ),
#         row=1,
#         col=1
#     )

#     fig.add_trace(
#         go.Image(z=np.zeros((10,10))),
#         row=1,
#         col=2
#     )
    
#     fig.update_layout(
#         scene=dict(
#             aspectmode='cube',
#             camera=dict(
#                 eye=dict(x=1.5, y=1.5, z=1.5)  # Initial camera position
#             )
#         ),
#         autosize=True,
#         margin=dict(l=0, r=0, b=0, t=0),
#     )

#     fig.update_layout(
#         updatemenus=[dict(
#             type="buttons",
#             buttons=[dict(label="Play",
#                           method="animate",
#                           args=[None])],
#         )],
#         sliders=[{
#             "steps": [
#                 {
#                     "method": "animate",
#                     "args": [[str(k)], dict(mode="immediate", frame=dict(duration=animation_duration, redraw=True))],
#                     "label": str(k)
#                 }
#                 for k in range(0, NOF)
#             ]
#         }]
#     )

#     animation_frames = []

#     for frame in range(1, NOF + 1):
#         points = np.load(output_data_location + f"frame{frame}" + ".npy")

#         frame_data = dict(
#             type="scatter3d",
#             x=points[0],
#             y=points[1],
#             z=points[2],
#         )

#         img_path = "animations/" + input.value + "animation/" + f"{frame:04d}.png"
#         img = im.open(img_path)
#         image_data = go.Image(z=img)


#         animation_frames.append(go.Frame(data = [frame_data, image_data], name=str(len(animation_frames))))

#     fig.frames = animation_frames

#     initial_frame = 0
#     fig.update(data=fig.frames[initial_frame].data)

#     fig.show()

#     if save_video:
#         images = []
#         for frame in fig.frames:
#             fig.update(data=frame.data)
#             img_bytes = fig.to_image(format="jpg")
#             images.append(imageio.imread(img_bytes))

#         imageio.mimsave("output/videos/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.mp4", images, fps=1)
    
#     if save_html:
#         fig.write_html("output/interactive_html/" + f"{datetime.today().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}.html", auto_play=False)
