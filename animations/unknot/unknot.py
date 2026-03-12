import bpy

NOF = 100
translation_y = 1

bpy.ops.curve.primitive_bezier_circle_add(
    radius=1.0,
    location=(0, 0, 0),
    enter_editmode=False,
)

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.curve.select_all(action='SELECT')
bpy.ops.curve.subdivide(number_cuts=1)
bpy.ops.object.mode_set(mode='OBJECT')

unknot = bpy.context.active_object
unknot_spline = unknot.data.splines[0]
unknot_spline.resolution_u = 25

control_point = unknot.data.splines[0].bezier_points[2]

control_point_left = unknot.data.splines[0].bezier_points[1]
control_point_right = unknot.data.splines[0].bezier_points[3]

control_point_left.handle_left_type = "ALIGNED"
control_point_left.handle_right_type = "ALIGNED"
control_point_right.handle_left_type = "ALIGNED"
control_point_right.handle_right_type = "ALIGNED"


bpy.context.scene.frame_set(0)
control_point.keyframe_insert(data_path="co", index=-1)
control_point.keyframe_insert(data_path="handle_left", index=-1)
control_point.keyframe_insert(data_path="handle_right", index=-1)

bpy.context.scene.frame_set(NOF)

new_y = control_point.co.y - translation_y
control_point.co = (control_point.co.x, new_y, control_point.co.z)

control_point.keyframe_insert(data_path="co", index=-1)
control_point.keyframe_insert(data_path="handle_left", index=-1)
control_point.keyframe_insert(data_path="handle_right", index=-1)

depsgraph = bpy.context.evaluated_depsgraph_get()    
