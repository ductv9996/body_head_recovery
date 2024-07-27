import bpy
import bmesh
import json

context = bpy.context
ob = context.edit_object
me = ob.data

bm = bmesh.from_edit_mesh(me)
# list of selected faces
selfaces = [f.index for f in bm.faces if f.select]
print(len(selfaces))

with open('/home/duc/Desktop/parts_measure/arm_faces.json', 'w') as file:
	json.dump(selfaces, file)