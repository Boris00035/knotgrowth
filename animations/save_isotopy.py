import bpy
import os
import numpy as np


NOF = 40


object_to_save = "unknot/twist/"
selected_object = bpy.data.objects.get("twist")

location_to_save = r"/home/boris/Documents/Universiteit_Utrecht/BscScriptie/knotgrowth/animations/" + object_to_save
#location_to_save = r"E:/Documents/Universiteit_Utrecht/BscScriptie/knotgrowth/src/knotgrowth/animations/" + object_to_save

depsgraph = bpy.context.evaluated_depsgraph_get()

for frame in range(1, NOF + 1):
    bpy.context.scene.frame_set(frame)
    eval_curve = selected_object.evaluated_get(depsgraph)
    mesh = eval_curve.to_mesh()

    points = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", points)
    points = points.reshape(-1, 3)

    np.save(location_to_save + f"frame{frame}" + ".npy", points)