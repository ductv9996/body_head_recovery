import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy import spatial

from scipy.spatial import ConvexHull
# import open3d as o3d

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(ROOT_DIR + '/part_measure_mesh/body_s_faces.json') as fb:
    smpl_idx_body_faces = json.load(fb)
# with open(ROOT_DIR + '/part_measure_mesh/right_arm_faces.json') as fa:
#     smpl_idx_rarm_faces = json.load(fa)
with open(ROOT_DIR + '/part_measure_mesh/right_leg_faces.json') as fl:
    smpl_idx_rleg_faces = json.load(fl)
# with open(ROOT_DIR + '/part_measure_mesh/ngang_vai_faces2.json') as fff4:
#     smpl_idx_ngang_vai_faces = json.load(fff4)
with open(ROOT_DIR + '/part_measure_mesh/dau_co_faces.json') as fff5:
    smpl_idx_dau_co_faces = json.load(fff5)
# with open(ROOT_DIR + '/part_measure_mesh/dai_chan_1_faces.json') as fff6:
#     smpl_idx_dai_chan_faces = json.load(fff6)
with open(ROOT_DIR + '/part_measure_mesh/arm_faces.json') as fff7:
    smpl_idx_arm_faces = json.load(fff7)
# with open(ROOT_DIR + '/part_measure_mesh/lower_arm_faces.json') as fff8:
#     smpl_idx_lower_arm_faces = json.load(fff8)
# with open(ROOT_DIR + '/part_measure_mesh/cao_than_faces.json') as fff9:
#     smpl_idx_cao_than_faces = json.load(fff9)
# with open(ROOT_DIR + '/part_measure_mesh/kich_sau_faces.json') as fff10:
#     smpl_idx_kich_sau_faces = json.load(fff10)
# with open(ROOT_DIR + '/part_measure_mesh/kich_truoc_faces.json') as fff11:
#     smpl_idx_kich_truoc_faces = json.load(fff11)
# with open(ROOT_DIR + '/part_measure_mesh/nach_phai_faces.json') as fff12:
#     smpl_idx_nach_phai_faces = json.load(fff12)
# with open(ROOT_DIR + '/part_measure_mesh/bnecktowaist_vert.json', 'r') as fbntw:
#     Bntw = json.load(fbntw)
with open(ROOT_DIR + '/part_measure_mesh/vong_bap_tay.json', 'r') as fbt:
    Bt = json.load(fbt)
# with open(ROOT_DIR + '/part_measure_mesh/day_quan.json', 'r') as fdq:
#     Dq = json.load(fdq)


info_measurements = {
    'shoulder_to_crotch':
        [
            828,
            3149,
        ],
    'back_length':
        [
            828,
            3021,
        ],
    'chest_circumference':
        {
            'position': 8384, # smplx
            'normal': (0, 1, 0),
        },
    'waist_circumference':
        {
            'position': 5939, # smplx
            'normal': (0, 1, 0),
        },
    'upper_waist_circumference':
        {
            'position': 3852, #smplx
            'normal': (0, 1, 0),
        },
    'pelvis_circumference':
        {
            'position': 8388, #smplx
            'normal': (0, 1, 0),
        },
    'bicep_circumference':
        {
            'position': 6181, #smplx
            'normal': None,
        },
    'forearm_circumference':
        {
            'position': 6955, #smplx
            'normal': None,
        },
    'arm_length':
        [
            8321,
            7558,
        ],
    'inside_leg_length':
        [
            1160,
            3458,
        ],
    'outside_leg_length':
        {
            'position': 6375,
            'normal': None,
        },
    'thigh_circumference':
        {
            'position': 6339, #smplx
            'normal': (0, 1, 0),
        },
    'calf_circumference':
        {
            'position': 6842,#smplx
            'normal': (0, 1, 0),
        },
    'overall_height':
        [
            9011,
            8706,#smplx
        ],
    'shoulder_breadth':
        [
            4493,
            7184,#smplx
        ],
    'curve_shoulder_breadth':
        {
            'position': 1306,
            'normal': None,
        },
    'neck_width':
        [
            218,
            3730,
        ],
    'bust_height':
        [
            3042,
            3458,
        ],
    'waist_height':
        [
            3500,
            3458,
        ],
    'hip_height':
        [
            3119,
            3458,
        ],
    'back_neck_height':
        [
            828,
            3458,
        ],
    'knee_height':
        [
            1046,
            3458,
        ],
    'inseam':
        [
            1160,
            3327,
        ],
    'back_neck_to_waist':
        [

        ],
    'back_neck_to_wrist':
        [
            828,
            5342,
            5128,
            5568,
        ],
    'head_circumference':
        {
            'position': 9238,#smplx
            'normal': (0, 1, 0),
        },
    'neck_circumference':
        {
            'position': 8938,##smplx,
            'normal': (0, 1, 0),
        },
    'wrist_circumference':
        {
            'position': 7580,#smplx
            'normal': None,
        },
    'ankle_circumference':
        {
            'position': 8571,
            'normal': (0, 1, 0),
        },
    'chest_width':
        [
            4132,
            644,
        ],
    'waist_width':
        [
            4121,
            631,
        ],
    'pelvis_width':
        [
            4297,
            809,
        ],
    'finger_to_finger':
        [
            5907,
            2446,
        ],
    'wrist_to_shoulder':
        [
            4132,
            5384,
        ],
    'wrist_to_wrist':
        [
            1924,
            5384,
        ],
    'inseam_width':
        [
            4801,
            1321,
        ],
}

def _HeightWeightfromBeta(betas_matrix, a, b, _inv_A, B):
    betas_matrix = torch.reshape(betas_matrix, (-1, 1))
    _vRoot_h = torch.mm(_inv_A, (betas_matrix - B))
    height = _vRoot_h[0, 0]
    vRoot = _vRoot_h[1, 0]
    v = vRoot ** 3
    weight = v * b + a

    return weight

def equation_plane(po1, po2, po3):
    a1 = po2[0] - po1[0]
    b1 = po2[1] - po1[1]
    c1 = po2[2] - po1[2]
    a2 = po3[0] - po1[0]
    b2 = po3[1] - po1[1]
    c2 = po3[2] - po1[2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    # d = (- a * po1[0] - b * po1[1] - c * po1[2])
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector ** 2))

    return normal_vector

def measure_len_mesh(po1, idx_faces, dir, vertices, faces, vis, po2=None, po3=None, norm_vector=None):
    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces[idx_faces], process=False)
    if norm_vector is None:
        normal_vector = equation_plane(po1, po2, po3)
    else:
        norm_vector = np.array(norm_vector)
        normal_vector = norm_vector / np.sqrt(np.sum(norm_vector ** 2))
    lines = trimesh.intersections.mesh_plane(mesh_tri, normal_vector, po1, return_faces=False)
    # print(lines.shape)
    points = np.zeros((lines.shape[0] * 2, 3))
    for i in range(lines.shape[0]):
        points[i] = lines[i][0]
        points[i + lines.shape[0]] = lines[i][1]
    # points = np.unique(points, axis=0)
    points2D = points.copy()

    v = np.array([1, -normal_vector[0] / normal_vector[1], 0])
    v = v / np.sqrt(np.sum(v ** 2))
    u = np.cross(normal_vector, v)
    M = np.array([v, normal_vector, u])
    for v in range(points.shape[0]):
        points2D[v] = np.dot(M, points[v])

    points2D = points2D[:, 0:3:2]

    hull = ConvexHull(points2D)
    boundary_points2D = points2D[hull.vertices]
    dist_mat = spatial.distance_matrix(boundary_points2D, boundary_points2D)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    dim = 0
    if dir == 1:  # do qua cac diem tu i--> j
        if vis == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(boundary_points2D[i:j + 1, 0], boundary_points2D[i:j + 1, 1], 'ro-')
            ax.scatter(points2D[:, 0], points2D[:, 1])
            plt.show()

        part_points = boundary_points2D[i:j + 1, :]
        for i in range(0, len(part_points) - 1):
            dim += np.linalg.norm(part_points[i + 1] - part_points[i])
        # ---------------------------------------------------------------------#

    if dir == 2:  # do qua cac diem j-->i
        if vis == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(boundary_points2D[j:, 0], boundary_points2D[j:, 1], 'ro-')
            ax.plot(boundary_points2D[0:i + 1, 0], boundary_points2D[0:i + 1, 1], 'ro-')
            ax.scatter(points2D[:, 0], points2D[:, 1])
            plt.show()

        part_points = np.append(boundary_points2D[j:, :], boundary_points2D[0:i + 1, :], axis=0)

        for i in range(0, len(part_points) - 1):
            dim += np.linalg.norm(part_points[i + 1] - part_points[i])
        # ---------------------------------------------------------------------#

    if dir == 3:  # do chu vi vong do
        if vis == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(boundary_points2D[:, 0], boundary_points2D[:, 1], 'ro-')
            ax.scatter(points2D[:, 0], points2D[:, 1])
            plt.show()

        part_points = boundary_points2D
        for i in range(-1, len(part_points) - 1):
            dim += np.linalg.norm(part_points[i + 1] - part_points[i])
        # ---------------------------------------------------------------------#

    # if vis == True:
    #     body_mesh = o3d.geometry.TriangleMesh()
    #     body_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #     body_mesh.triangles = o3d.utility.Vector3iVector(faces)
    #     body_mesh.compute_vertex_normals()
    #     body_mesh.paint_uniform_color([163 / 255, 156 / 255, 155 / 255])

    #     point_cloud = o3d.geometry.PointCloud()
    #     point_cloud.points = o3d.utility.Vector3dVector(points)
    #     point_cloud.paint_uniform_color([0 / 255, 0 / 255, 255 / 255])
    #     o3d.visualization.draw_geometries([body_mesh, point_cloud])

    return dim

def get_measure(smpl_vertices,
                vertices,
                faces,
                info_measurements,
                idx_body_faces,
                idx_dau_co_faces,
                idx_rleg_faces,
                idx_arm_faces,
                Bt,
                vis_measure=False):
    measures = dict()

    # measures['curve_shoulder_breadth'] = measure_len_mesh(
    #     po1=smpl_vertices[info_measurements['curve_shoulder_breadth']['position']],
    #     idx_faces=idx_ngang_vai_faces,
    #     dir=2,
    #     vertices=vertices,
    #     faces=faces,
    #     vis=vis_measure,
    #     po2=vertices[1862],
    #     po3=vertices[5325],
    #     norm_vector=info_measurements['curve_shoulder_breadth']['normal'],
    #     )
        
    # measures['arm_length'] = measure_len_mesh(
    #     po1=smpl_vertices[5325],
    #     idx_faces=idx_upper_arm_faces,
    #     dir=2,
    #     vertices=vertices,
    #     faces=faces,
    #     vis=vis_measure,
    #     # po2=vertices[5172], #beo
    #     po2=vertices[5290], #gay
    #     po3=vertices[5205], #5205
    #     ) + measure_len_mesh(
    #     po1=smpl_vertices[5205],
    #     idx_faces=idx_lower_arm_faces,
    #     dir=2,
    #     vertices=vertices,
    #     faces=faces,
    #     vis=vis_measure,
    #     po2=vertices[5400],
    #     po3=vertices[5567],
    #     ) + 0.015
    measures['arm_length'] = np.linalg.norm(smpl_vertices[info_measurements["arm_length"][0]] - smpl_vertices[info_measurements["arm_length"][1]])+ 0.06

    tmp = 0.
    for i in range(-1, len(Bt) - 1):
        tmp += np.linalg.norm(vertices[Bt[i + 1]] - vertices[Bt[i]])
    measures['bicep_circumference'] = tmp + 0.01

    measures['forearm_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['forearm_circumference']['position']],
        idx_faces=idx_arm_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        po2=vertices[6944], # smplx
        po3=vertices[7103],
        norm_vector=None,
        )
    
    measures['wrist_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['wrist_circumference']['position']],
        idx_faces=idx_arm_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        po2=vertices[7559], # smplx
        po3=vertices[7458],
        norm_vector=None,
        )

    measures['neck_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['neck_circumference']['position']],
        idx_faces=idx_dau_co_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        po2=vertices[3184], # smplx
        po3=vertices[5617],
        norm_vector=None,
        )
    
    measures['head_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['head_circumference']['position']],
        idx_faces=idx_dau_co_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        po2=None,
        po3=None,
        norm_vector=info_measurements['head_circumference']['normal'],
        )

    measures['chest_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['chest_circumference']['position']],
        idx_faces=idx_body_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['chest_circumference']['normal'],
        )
    
    measures['shoulder_breadth'] = np.linalg.norm(smpl_vertices[info_measurements["shoulder_breadth"][0]] - smpl_vertices[info_measurements["shoulder_breadth"][1]])
        
    # measures['upper_waist_circumference'] = measure_len_mesh(
    #     po1=smpl_vertices[info_measurements['upper_waist_circumference']['position']],
    #     idx_faces=idx_body_faces,
    #     dir=3,
    #     vertices=vertices,
    #     faces=faces,
    #     vis=vis_measure,
    #     norm_vector=info_measurements['upper_waist_circumference']['normal'],
    #     )
    
    # measures['outside_leg_length'] = measure_len_mesh(
    #         po1=vertices[1791],
    #         idx_faces=idx_dai_chan_faces,
    #         dir=1,
    #         vertices=vertices,
    #         faces=faces,
    #         vis=vis_measure,
    #         po2=vertices[815],
    #         po3=vertices[3328],
    #         norm_vector=None,
    #     )
    
    measures['waist_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['upper_waist_circumference']['position']],
        idx_faces=idx_body_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['upper_waist_circumference']['normal'],
        )
    
    measures['pelvis_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['pelvis_circumference']['position']],
        idx_faces=idx_body_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['pelvis_circumference']['normal'],
        )

    measures['thigh_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['thigh_circumference']['position']],
        idx_faces=idx_rleg_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['thigh_circumference']['normal'],
        )
        
    measures['calf_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['calf_circumference']['position']],
        idx_faces=idx_rleg_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['calf_circumference']['normal'],
        )

    measures['ankle_circumference'] = measure_len_mesh(
        po1=smpl_vertices[info_measurements['ankle_circumference']['position']],
        idx_faces=idx_rleg_faces,
        dir=3,
        vertices=vertices,
        faces=faces,
        vis=vis_measure,
        norm_vector=info_measurements['ankle_circumference']['normal'],
        )
    
    return measures


def measure_mesh(faces, np_verts):
    vis_measure = False

    measures = get_measure(smpl_vertices=np_verts,
                           vertices=np_verts,
                           faces=faces,
                           info_measurements=info_measurements,
                           idx_body_faces=smpl_idx_body_faces,
                           idx_dau_co_faces=smpl_idx_dau_co_faces,
                           idx_arm_faces=smpl_idx_arm_faces,
                           idx_rleg_faces=smpl_idx_rleg_faces,
                           Bt=Bt,
                           vis_measure=vis_measure,
                           )
    # measures = np.insert(measures, 2,
    #                      _HeightWeightfromBeta(betas_matrix=torch.tensor(betas, dtype=torch.float32).cpu().unsqueeze(0), a=a, b=b, _inv_A=_inv_A, B=B) / 100)
    return measures
