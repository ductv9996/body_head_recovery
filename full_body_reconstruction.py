import bpy
import os
import sys

import cv2
import numpy as np
import body_head_recovery.config as config
from PIL import Image

import torch
import mediapipe as mp
from body_head_recovery.utils.uv_utils import *
from body_head_recovery.utils.read_obj import read_obj_file
from body_head_recovery.optim.head_optim import HEAD_OPTIM
from body_head_recovery.bodyrecon.body_recon import body_from_image_params

from body_head_recovery.Color_Transfer.transfer_color import run_transfer
from body_head_recovery.Inpainting.head_inpaint import run_inpaint

import mathutils
from mathutils import Vector, Quaternion
from laplacian_pyramid_blend import LaplacianPyramidBlender
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')

def get_face_rect(img):
    results = config.face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection = results.detections[0]
    box = detection.location_data.relative_bounding_box

    height, width = img.shape[:2]
    left = int(box.xmin * width)
    top = int(box.ymin * height)
    right = int(left + int(box.width * width))
    bottom = int(top + int(box.height * height))
    return top, left, right, bottom

def get_face_landmarks(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    detection_result = config.face_detector.detect(image)

    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    idx = 0
    face_landmarks = face_landmarks_list[idx]

    np_lmks = np.zeros((len(face_landmarks), 2))
    # real_lmks = np.zeros((len(face_landmarks), 3))
    for i, landmark in enumerate(face_landmarks):
        x = int(landmark.x * input_img.shape[1])
        y = int(landmark.y * input_img.shape[0])
        np_lmks[i] = np.array([x, y])
        # real_lmks[i] = np.array([landmark.x, landmark.y, landmark.z])
    return np_lmks


def possion_blending(src, tar, mask_compose, kernel_size):

    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    mask_compose = cv2.erode(mask_compose, kernel, iterations=1) 
    monoMaskImage = cv2.split((mask_compose*255).astype(np.uint8))[0] # reducing the mask to a monochrome
    br = cv2.boundingRect(monoMaskImage) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)

    result = cv2.seamlessClone(src.astype(np.uint8), tar.astype(np.uint8),
                                        (mask_compose*255).astype(np.uint8), centerOfBR,
                                        cv2.NORMAL_CLONE)
    
    return result.astype(np.uint8)


def head_recon(gender, image_f, image_r, image_l):
    landmarks_f = get_face_landmarks(image_f)
    landmarks_f = torch.tensor(landmarks_f)
    croped_img_f, croped_lmks_f = crop_image(image_f, landmarks_f)
    
    landmarks_r = get_face_landmarks(image_r)
    landmarks_r = torch.tensor(landmarks_r)
    croped_img_r, croped_lmks_r = crop_image(image_r, landmarks_r)

    landmarks_l = get_face_landmarks(image_l)
    landmarks_l= torch.tensor(landmarks_l)
    croped_img_l, croped_lmks_l = crop_image(image_l, landmarks_l)

    vec_r = croped_lmks_r[93] - croped_lmks_r[1]
    vec_l = croped_lmks_r[323] - croped_lmks_r[1]

    if abs(vec_r[0]) < abs(vec_l[0]):
        swap_lmk = croped_lmks_l.clone()
        swap_img = croped_img_l.copy()
        croped_lmks_l = croped_lmks_r
        croped_img_l = croped_img_r
        croped_lmks_r = swap_lmk
        croped_img_r = swap_img

    # cv2.namedWindow("aa", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("aa", 1024, 1024)
    # img_f = croped_img_f.copy()
    # img_r = croped_img_r.copy()
    # img_l = croped_img_l.copy()
    # for p1 in croped_lmks_f:
    #     cv2.circle(img_f, (int(p1[0]), int(p1[1])), 2, (0, 0, 255), -1)
        
    # for p1 in croped_lmks_r:
    #     cv2.circle(img_r, (int(p1[0]), int(p1[1])), 2, (0, 255, 0), -1)
        
    # for p1 in croped_lmks_l:
    #     cv2.circle(img_l, (int(p1[0]), int(p1[1])), 2, (0, 255, 0), -1)
        
    # cv2.imshow("cc", img_l)
    # cv2.imshow("bb", img_r)
    # cv2.imshow("aa", img_f)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    head_optim = HEAD_OPTIM(gender=gender, vis=False)

    verts_show, verts_f, cam_f, verts_r, cam_r, verts_l, cam_l = head_optim(image_f=croped_img_f, landmarks_f=croped_lmks_f, 
                                                                image_r=croped_img_r, landmarks_r=croped_lmks_r, 
                                                                image_l=croped_img_l, landmarks_l=croped_lmks_l)
    # convert vertices to cm for display in mobile
    # verts_show = verts_show*100.
    
    texture_f = transfer_texture(body_verts=verts_f, cam=cam_f, croped_img=croped_img_f)
    texture_r = transfer_texture(body_verts=verts_r, cam=cam_r, croped_img=croped_img_r)
    texture_l = transfer_texture(body_verts=verts_l, cam=cam_l, croped_img=croped_img_l)

    # texture_lr = possion_blending(src=texture_l, tar=texture_r, mask_compose=config.uv_face_left, kernel_size=1)
    texture_lr = config.uv_face_left*texture_l + (1-config.uv_face_left)*texture_r
    # final_texture = possion_blending(src=texture_f, tar=texture_lr, mask_compose=config.uv_face_front, kernel_size=5)
    blender = LaplacianPyramidBlender()
    final_texture = blender(texture_f, texture_lr, config.uv_face_front*255)
    final_texture = config.head_mask*final_texture + (1-config.head_mask)*255
    
    return verts_show.cpu().squeeze(), final_texture.astype(np.uint8), texture_f


# def merger_body_head(gender, body_verts, image_f, image_r, image_l):

#     if image_f.shape[1] > 720:
#         image_f = cv2.resize(image_f, (720, int(720*image_f.shape[0]/image_f.shape[1])))
#     if image_r.shape[1] > 720:
#         image_r = cv2.resize(image_r, (720, int(720*image_r.shape[0]/image_r.shape[1])))
#     if image_l.shape[1] > 720:
#         image_l = cv2.resize(image_l, (720, int(720*image_l.shape[0]/image_l.shape[1])))

    
#     head_verts, final_texture = head_recon(gender=gender, image_f=image_f, image_r=image_r, image_l=image_l)

#     # process body skin
#     src_img_cv = cv2.imread(f"body_head_recovery/data/texture/{gender}_vang_hair.png")
#     tar_img_cv = final_texture.copy()[272:272+410, 409:409+410]

#     transfered_img = run_transfer(src_img_cv=src_img_cv, tar_img_cv=tar_img_cv)

#     transfered_texture = config.full_face_mask*final_texture + (1-config.full_face_mask)*transfered_img
#     inpainted_texture = run_inpaint(orig_img=transfered_texture.astype(np.uint8))

#     # process inner_wear
#     if gender =="male":
#         inner_wear_img = config.innerwear_male
#         inner_wear_mask = config.innerwear_mask_male
#     else:
#         inner_wear_img = config.innerwear_female
#         inner_wear_mask = config.innerwear_mask_female
    
#     complete_texture = inner_wear_mask*inner_wear_img + (1-inner_wear_mask)*inpainted_texture

#     scaleo, Ro, to = compute_similarity_transform_torch(head_verts[config.lowest_head_smplx_idx], body_verts[config.lowest_head_smplx_idx])
    
#     trans_body_vert = scaleo * Ro.mm(head_verts.T) + to
#     trans_body_vert = trans_body_vert.T

#     body_verts[config.smplx2head_idx] = trans_body_vert[config.smplx2head_idx]

#     return body_verts.cpu(), complete_texture.astype(np.uint8)


def run_head(gender, image_f, image_r, image_l):

    if image_f.shape[1] > 720:
        image_f = cv2.resize(image_f, (720, int(720*image_f.shape[0]/image_f.shape[1])))
    if image_r.shape[1] > 720:
        image_r = cv2.resize(image_r, (720, int(720*image_r.shape[0]/image_r.shape[1])))
    if image_l.shape[1] > 720:
        image_l = cv2.resize(image_l, (720, int(720*image_l.shape[0]/image_l.shape[1])))

    
    head_verts, final_texture, texture_f = head_recon(gender=gender, image_f=image_f, image_r=image_r, image_l=image_l)

    # process body skin
    src_img_cv = cv2.imread(f"{config.texture_dir}/{gender}_vang_hair.png")
    # tar_img_cv = texture_f.copy()[232:232+410, 402:402+410]
    tar_img_cv = texture_f.copy()[144:144+408, 424:424+371]

    # transfered_img_ref = run_transfer(src_img_cv=src_img_cv.copy(), tar_img_cv=tar_img_cv)
    
    # mean_vals = cv2.mean(transfered_img_ref[1020:1020+250, 510:510+250])[:3]
    # image_ref = np.zeros((128, 128, 3), np.uint8)
    # image_ref[:] = (int(mean_vals[0]), int(mean_vals[1]), int(mean_vals[2]))
    transfered_img = run_transfer(src_img_cv=src_img_cv.copy(), tar_img_cv=tar_img_cv)

    transfered_texture = config.full_face_mask*final_texture + (1-config.full_face_mask)*transfered_img

    inpainted_texture = run_inpaint(orig_img=transfered_texture.astype(np.uint8))
    smooth_inpainted_texture = possion_blending(inpainted_texture.astype(np.uint8), transfered_img.astype(np.uint8), config.extend_face_mask, kernel_size=5)

    # process inner_wear
    if gender =="male":
        inner_wear_img = config.innerwear_male
        inner_wear_mask = config.innerwear_mask_male
        complete_texture = inner_wear_mask*inner_wear_img + (1-inner_wear_mask)*inpainted_texture
    else:
        inner_wear_img = config.innerwear_female
        inner_wear_mask = config.innerwear_mask_female
        complete_texture = inner_wear_mask*inner_wear_img + (1-inner_wear_mask)*smooth_inpainted_texture
    
    gray_mask_eye = cv2.cvtColor(config.eye_mask, cv2.COLOR_BGR2GRAY)
    ret_eye, thresh_mask_eye = cv2.threshold(gray_mask_eye, 127, 255, cv2.THRESH_BINARY)
    mask_eye = np.where(thresh_mask_eye == 255)
    complete_texture[mask_eye] = config.eye_brown[mask_eye]

    return head_verts.cpu(), complete_texture.astype(np.uint8)


def merger_body_head(body_verts, head_verts):

    body_verts_only = body_verts[:10475]
    body_joints = torch.tensor(body_verts[10475:], dtype=torch.float32)

    body_verts_only = torch.tensor(body_verts_only, dtype=torch.float32)
    head_verts = torch.tensor(head_verts, dtype=torch.float32)

    scaleo, Ro, to = compute_similarity_transform_torch(head_verts[config.lowest_head_smplx_idx], body_verts_only[config.lowest_head_smplx_idx])
    
    trans_body_vert = scaleo * Ro.mm(head_verts.T) + to
    trans_body_vert = trans_body_vert.T

    body_verts_only[config.smplx2head_idx] = trans_body_vert[config.smplx2head_idx]

    body_verts_joints = torch.cat([body_verts_only, body_joints])

    return body_verts_joints.cpu()


# def merger_body_hair(body_head_verts, texture, hair_input_path, avatar_output_path):

#     ply_data = trimesh.load(hair_input_path)
#     hair_verts = torch.tensor(ply_data.vertices, dtype=torch.float32)

#     # R_x = torch.tensor([[1.0, 0.0, 0.0],
#     #                     [0., 0.0, -1.0],
#     #                     [0.0, 1.0, 0.0]])

#     # hair_verts_x = torch.mm(R_x, hair_verts.T)
#     # hair_verts_x = hair_verts_x.T

#     head_hair_verts = read_obj_file("body_head_recovery/models/head_model.obj")
#     head_hair_verts = torch.tensor(head_hair_verts, dtype=torch.float32)
#     scaleo, Ro, to = compute_similarity_transform_torch(head_hair_verts[config.hair_head_idx], body_head_verts[config.body_head_idx])
#     trans_hair_vert = scaleo * Ro.mm(hair_verts.T) + to
#     trans_hair_vert = trans_hair_vert.T

#     ply_data.vertices = trans_hair_vert

#     out_path = f"temp/"
#     if not os.path.exists(out_path):
#         # Create the directory
#         os.makedirs(out_path)

#     #save hair ply
#     full_body_glb = trimesh.load(f"body_head_recovery/data/body_temp/body_temp.glb")
#     material_body = trimesh.visual.texture.PBRMaterial(baseColorTexture=Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)),
#                                                        roughnessFactor=0.9036020036098448,
#                                                        metallicFactor=0.0, doubleSided=True)
#     full_body_glb.geometry['body_temp.obj'].visual.material = material_body

#     map_body = config.idx_map_glb_obj.numpy().astype(np.int32)
#     full_body_glb.geometry['body_temp.obj'].visual.mesh.vertices = body_head_verts.cpu().numpy()[map_body]

#     full_body_glb.add_geometry(ply_data)
#     full_body_glb.export(avatar_output_path)
#     # # Save body verts 
#     # cv2.imwrite(out_path + f"final_texture.png", texture)
#     # with open(f'body_head_recovery/data/body_temp/body_temp.obj', 'r') as ft:
#     #     merge_lines = ft.readlines()
#     # with open(out_path + f"body_head.obj", 'w') as fm:
#     #     fm.write(merge_lines[0])
#     #     fm.write(merge_lines[1])
#     #     fm.write(merge_lines[2])
#     #     for v in body_verts:
#     #         fm.write(f"v {v[0]} {v[1]} {v[2]}\n")

#     #     for f in merge_lines[10479:]:
#     #         fm.write(f)

#     # shutil.copyfile("body_head_recovery/data/body_temp/body_temp.mtl", out_path + f"body_temp.mtl")

def merger_body_hair(body_head_verts, texture, hair_result, avatar_output_path):

    body_joint_np = (body_head_verts.clone()[10475:]).numpy()
    body_head_verts = body_head_verts.clone()[:10475]

    # process hair 
    hair_verts = np.asarray(hair_result['pc_all_valid'])
    lines = np.asarray(hair_result['lines'])
    colors = hair_result["colors"]

    hair_verts = torch.tensor(hair_verts, dtype=torch.float32)
    hair_faces = []
    for id_f in range(lines.shape[0] -4):
        if lines[id_f][1] == lines[id_f +1][0]:
            hair_faces.append([lines[id_f][0], lines[id_f][1], lines[id_f+1][1]])
    hair_faces = np.array(hair_faces, dtype=np.int32)

    head_hair_verts = read_obj_file(config.hairstep_head_temp_path)
    head_hair_verts = torch.tensor(head_hair_verts, dtype=torch.float32)
    scaleo, Ro, to = compute_similarity_transform_torch(head_hair_verts[config.hair_head_idx], body_head_verts[config.body_head_idx])
    trans_hair_vert = scaleo * Ro.mm(hair_verts.T) + to
    trans_hair_vert = trans_hair_vert.T.numpy()

    trans_head_hair_verts = scaleo * Ro.mm(head_hair_verts.T) + to
    trans_head_hair_verts = trans_head_hair_verts.T.numpy()

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    trans_hair_vert = trans_hair_vert.dot(rotation_matrix)

    bpy_refresh()
    bpy.ops.import_scene.gltf(filepath=config.default_hair_path["male"])
    hair_obj = process_hair(np_verts=trans_hair_vert, np_faces=hair_faces, colors=colors)
    human_fbx = process_body_bpy(np_body_verts=body_head_verts.numpy(),np_body_joints=body_joint_np, texture_cv=texture)
    transfer_weight(source_fbx=human_fbx, target_name="Hair")
    # Select the object to export
    bpy.ops.object.select_all(action="SELECT")

    # Export the mesh to .glb format
    bpy.ops.export_scene.gltf(filepath=avatar_output_path, export_format='GLB', use_selection=True)
    bpy_refresh()


def merger_body_hair_temp(body_head_verts, texture, hair_glb_path, hair_color, avatar_output_path):

    body_joint_np = (body_head_verts.clone()[10475:]).numpy()
    body_head_verts = body_head_verts.clone()[:10475]

    ## process default hair
    body_default_verts = torch.tensor(read_obj_file(config.body_temp_obj_path), dtype=torch.float32)
    scaleo, Ro, to = compute_similarity_transform_torch(body_default_verts[config.body_head_idx], body_head_verts[config.body_head_idx])

    bpy_refresh()
    bpy.ops.import_scene.gltf(filepath=hair_glb_path)
    process_default_hair(scale=scaleo.numpy(),rotation_matrix=Ro.numpy(), translation=to.numpy(), colors=None)
    human_fbx = process_body_bpy(np_body_verts=body_head_verts.numpy(),np_body_joints=body_joint_np, texture_cv=texture)

    # rig hair
    Hair_Model = bpy.data.objects['Hair_Model']
    # Get all children of the object
    children = Hair_Model.children
    # Loop through children and print their names
    for child in children:
        transfer_weight(source_fbx=human_fbx, target_name=child.name)
    # Select the object to export
    bpy.ops.object.select_all(action="SELECT")
    # Export the mesh to .glb format
    bpy.ops.export_scene.gltf(filepath=avatar_output_path, export_format='GLB', use_selection=True)
    bpy_refresh()


def process_body_bpy(np_body_verts, np_body_joints, texture_cv):

    model_path = config.body_temp_fbx_path

    # flip image corresponds with texture in blender
    texture_cv = np.flipud(texture_cv)
    texture_cv = cv2.cvtColor(texture_cv, cv2.COLOR_BGR2RGBA)  # Convert from BGR to RGB

    bpy.ops.import_scene.fbx(filepath=model_path, ignore_leaf_bones=False, global_scale=1.0)
    bpy.ops.object.select_all(action="DESELECT")
    
    human_fbx = bpy.data.objects[f"SMPLX-male"]
    human_fbx.select_set(True)
    bpy.context.view_layer.objects.active = human_fbx

    bpy.ops.object.mode_set(mode="EDIT")

    # update joint location
    
    for index in range(config.NUM_SMPLX_JOINTS):
        bone = human_fbx.data.edit_bones[config.SMPLX_JOINT_NAMES[index]]
        # bone.head = (0.0, 0.0, 0.0)
        # bone.tail = (0.0, 0.0, 0.1)

        # # Convert SMPL-X joint locations to Blender joint locations
        # joint_location_smplx = np_body_joints[index]
        # bone_start = Vector((joint_location_smplx[0], -joint_location_smplx[2], joint_location_smplx[1]) )
        # bone.translate(bone_start)

        head = np.array([np_body_joints[index][0], -np_body_joints[index][2], np_body_joints[index][1]])
        tail = head - np.array(bone.head[:]) + np.array(bone.tail[:])
        bone.head = head
        bone.tail = tail

    v = 0
    for vert in human_fbx.children[0].data.vertices:
        # vert.co = 100 * body_mesh[v]
        vert.co[0] = np_body_verts[v][0]
        vert.co[1] = -np_body_verts[v][2]
        vert.co[2] = np_body_verts[v][1]
        v = v + 1

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

    human_obj = human_fbx.children[0]
    bpy.context.view_layer.objects.active = human_obj
    ob = bpy.context.view_layer.objects.active

    # Get the existing material (assuming there's only one material on the object)
    material = ob.data.materials[0]

    # Enable 'Use nodes' for the material if not already enabled
    if not material.use_nodes:
        material.use_nodes = True

    # Get the material's node tree
    nodes = material.node_tree.nodes

    # Find the Principled BSDF node
    principled_bsdf = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break

    if principled_bsdf is None:
        raise Exception("No Principled BSDF shader found in the material")

    # Create a new Image Texture node
    texture_node = nodes.new(type='ShaderNodeTexImage')

    # Create a Blender image and fill it with the OpenCV image data
    image_height, image_width, _ = texture_cv.shape
    texture_image = bpy.data.images.new(name="TextureImage", width=image_width, height=image_height)

    # Flatten the image array and assign it to Blender image pixels
    texture_image.pixels = (texture_cv / 255.0).flatten()
    # Save the Blender image to a file
    texture_image.filepath_raw = "//static/humanTexture.png"
    texture_image.file_format = 'PNG'
    texture_image.save()

    # Assign the Blender image to the texture node
    texture_node.image = texture_image

    # Link the texture node to the Base Color input of the Principled BSDF node
    links = material.node_tree.links
    links.new(texture_node.outputs['Color'], principled_bsdf.inputs['Base Color'])

    bpy.ops.file.pack_all()

    return human_fbx


def transfer_weight(source_fbx, target_name):
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT") #deselecting everything
    bpy.data.objects[target_name].select_set(True) #selecting target
    source_fbx.children[0].select_set(True) #selecting source
#    
    bpy.context.view_layer.objects.active = bpy.data.objects[target_name] #setting target as active
#   
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.ops.object.data_transfer(use_reverse_transfer=True, data_type="VGROUP_WEIGHTS", 
        layers_select_src="NAME", layers_select_dst="ALL") #transferring weights
    bpy.ops.object.mode_set(mode="OBJECT")
        

    bpy.ops.object.select_all(action="DESELECT")
    bpy.data.objects[target_name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[target_name]
    bpy.ops.object.modifier_add(type="ARMATURE")
    bpy.context.object.modifiers["Armature"].object = source_fbx


def bpy_refresh():
    for item in bpy.data.objects:
        bpy.data.objects.remove(item, do_unlink=True)

    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item, do_unlink=True)

    for item in bpy.data.materials:
        bpy.data.materials.remove(item, do_unlink=True)

    for item in bpy.data.textures:
        bpy.data.textures.remove(item, do_unlink=True)

    for item in bpy.data.images:
        if item.name == "Render Result": continue
        bpy.data.images.remove(item, do_unlink=True)

    for item in bpy.data.lights:
        bpy.data.lights.remove(item, do_unlink=True)

    for item in bpy.data.cameras:
        bpy.data.cameras.remove(item, do_unlink=True)
        # Clear existing data
    bpy.ops.wm.read_factory_settings(use_empty=True)


def process_default_hair(scale, rotation_matrix, translation, colors):
    # Select the object to transform
    hair_model = bpy.data.objects["Hair_Model"]


    # Your provided 3x3 rotation matrix
    rotation_matrix_3x3 = mathutils.Matrix((
        (rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2]),
        (rotation_matrix[1,0],  rotation_matrix[1,1], rotation_matrix[1,2]),
        (rotation_matrix[2,0],  rotation_matrix[2,1],  rotation_matrix[2,2])
    ))

    # Convert to 4x4 rotation matrix by adding the extra row and column
    rotation_matrix_4x4 = rotation_matrix_3x3.to_4x4()

    # Scale matrix (example: scale by 2 in all axes)
    scale_matrix = mathutils.Matrix.Scale(scale, 4)

    # Translation vector (example: move by (3, 5, 2))
    translation_vector = mathutils.Vector((translation[0], translation[1], translation[2]))

    # Create the translation matrix
    translation_matrix = mathutils.Matrix.Translation(translation_vector)

    # Conversion matrix to switch from Y-up to Z-up (Blender's system)
    rotx_matrix = mathutils.Matrix((
        (1, 0, 0, 0),  # X stays the same
        (0, 0, 1, 0),  # Y becomes Z
        (0, -1, 0, 0),  # Z becomes Y
        (0, 0, 0, 1)   # Keep the homogeneous coordinate
    ))
    revert_rotx_matrix = mathutils.Matrix((
        (1, 0, 0, 0),  # X stays the same
        (0, 0, -1, 0),  # Y becomes Z
        (0, 1, 0, 0),  # Z becomes Y
        (0, 0, 0, 1)   # Keep the homogeneous coordinate
    ))

    # Combine transformations (translation * rotation * scale) with conversion
    transform_matrix = translation_matrix @ scale_matrix @ rotation_matrix_4x4 @ rotx_matrix 


    # Apply transformation to object
    hair_model.matrix_world = hair_model.matrix_world @ transform_matrix
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action='DESELECT')

    hair_model.matrix_world = hair_model.matrix_world @ revert_rotx_matrix
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def process_hair(np_verts, np_faces, colors):
    # Create mesh and object
    hair_mesh = bpy.data.meshes.new(name='HairMesh')
    hair_obj = bpy.data.objects.new(name='Hair', object_data=hair_mesh)

    # Link object to collection
    bpy.context.collection.objects.link(hair_obj)

    # Create mesh from given vertices and faces
    hair_mesh.from_pydata(np_verts.tolist(), [], np_faces.tolist())
    hair_mesh.update()

    # Create material
    material = bpy.data.materials.new(name="NMaterial")
    if colors[0][0] > 1. or colors[0][1] >1 or colors[0][2] > 1:
        material.diffuse_color = (colors[0][0]/255., colors[0][1]/255., colors[0][2]/255., 1.0)
    else:
        material.diffuse_color = (colors[0][0], colors[0][1], colors[0][2], 1.0)

    # Assign material to object
    if hair_obj.data.materials:
        hair_obj.data.materials[0] = material
    else:
        hair_obj.data.materials.append(material)

    # Add solidify modifier to make the mesh thicker
    solidify_modifier = hair_obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify_modifier.thickness = 0.001

    # Apply the solidify modifier
    bpy.context.view_layer.objects.active = hair_obj
    bpy.ops.object.modifier_apply(modifier=solidify_modifier.name)


    # Add decimate modifier to reduce polygon count
    decimate_modifier = hair_obj.modifiers.new(name="Decimate", type='DECIMATE')
    decimate_modifier.ratio = 0.5  # Adjust the ratio as needed (0.0 to 1.0)

    # Apply the decimate modifier
    bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)

    return hair_obj
