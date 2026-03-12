import knotgrowth.animations as anim 
from knotgrowth.animations import possible_inputs

def main():
    grid_size = 70
    num_iterations = 50
    num_cell_segments = 30 # How many colours we consider, without the outside
    num_frames = 40

    # todo 1: Make this plot all the colours, by adding them to seperate traces
    # todo 2: Make a growth animation
    # anim.create_boundary_animation(grid_size, num_iterations, num_frames, num_cell_segments, possible_inputs.unknot_twist)
    anim.create_boundary_animation_sp_from_files(grid_size, possible_inputs.unknot_dent)
    return



if __name__ == "__main__":
    main()
