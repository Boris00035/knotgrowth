import knotgrowth.animations as anim 
from knotgrowth.animations import possible_inputs

def main():
    grid_size = 70
    num_iterations = 50
    num_labels = 31
    num_frames = 1 # Must match the amount of frames from the input animation

    # todo 1: Make this plot all the colours, by adding them to seperate traces
    # todo 2: Make a growth animation
    
    anim.generate_grids_after_growth(grid_size, num_iterations, num_frames, num_labels, possible_inputs.unknot_circle, save_boundary=True, save_grid=True)
    anim.view_boundary_animation_3d(possible_inputs.unknot_double_twist)
    # anim.view_boundary_animation_sp(possible_inputs.unknot_dent, grid_size)
    return



if __name__ == "__main__":
    main()
