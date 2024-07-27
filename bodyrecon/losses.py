import torch
# from chamferdist.chamfer import knn_points
from pytorch3d.ops.knn import knn_points
import os
import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2


def init_beta2(joints2D):
    a = abs(joints2D[:, 1][0][1] - joints2D[:, 8][0][1])
    b = abs(joints2D[:, 8][0][1] - joints2D[:, 11][0][1])
    k = a / b

    beta2 = 79.365079365 * k - 44.238095238
    if beta2 > 2:
        beta2 = 2
    if beta2 < -2:
        beta2 = -2

    return beta2


def _HeightWeightfromBeta(betas_matrix, a, b, _inv_A, B):
    betas_matrix = torch.reshape(betas_matrix, (-1, 1))
    _vRoot_h = torch.mm(_inv_A, (betas_matrix - B))
    height = _vRoot_h[0, 0]
    vRoot = _vRoot_h[1, 0]
    v = vRoot ** 3
    weight = v * b + a

    return height, weight


def _BetafromHeightWeight(height, weight, a, b, A, B):
    v = (weight - a) / b
    vRoot = v**(1./3)
    _vRoot_h = torch.tensor([height, vRoot], device=height.device).unsqueeze(1)

    _betasMatrix = torch.mm(A, _vRoot_h) + B

    return torch.transpose(_betasMatrix, 0, 1)


def _BetaFromRegressor(height, weight, armspan, inseam, inseamWidth, wristToShoulder, a, b, A, B):
    v = (weight - a) / b
    vRoot = v ** (1. / 3)

    _vRoot_h6 = torch.tensor([armspan, height, inseamWidth, inseam, wristToShoulder, vRoot], device=A.device).unsqueeze(1)
    _betasMatrix = torch.mm(A, _vRoot_h6) + B

    return torch.transpose(_betasMatrix, 0, 1)


def calcul_k_weight(gender, BMI, age):

    if gender == 'male':

        if BMI <= 18.5:
            k_weight = -0.5
        elif 18.5 < BMI <= 23:
            k_weight = -2.5
        elif 23 < BMI <= 25:
            k_weight = -3.5
        elif 25 < BMI <= 29.9:
            k_weight = -4.
        elif 29.9 < BMI:
            k_weight = -5.5
    if gender == 'female':
        if BMI <= 18.5:
            # k_weight = -0.741 * BMI + 13.708
            k_weight = 0.
        elif 18.5 < BMI <= 23:
            # k_weight = -0.397 * BMI + 7.495
            k_weight = -2.5
        elif 23 < BMI <= 25:
            # k_weight = -0.398 * BMI + 7.374
            k_weight = -3.5
        elif 25 < BMI <= 29.9:
            # k_weight = -0.743 * BMI + 16.57
            k_weight = -5.
        elif BMI > 29.9:
            k_weight = -5.5
    return k_weight


def body_shape_loss_1(model_vertices, human_vertices):
    x_nn = knn_points(model_vertices, human_vertices, lengths1=None, lengths2=None, K=1)
    dist = x_nn.dists[..., 0]

    loss = torch.mean(dist)

    return loss


def nn_points_loss(template_vertices, scan_vertices):
    x_nn = knn_points(template_vertices, scan_vertices, lengths1=None, lengths2=None, K=1)
    dist = x_nn.dists[..., 0]

    loss = torch.mean(torch.sqrt(dist))
    return loss


def smooth_surface_loss(input_A_matrix, input_edges):
    edges_vert_1 = input_edges[:, 0]
    edges_vert_2 = input_edges[:, 1]

    A_Frobe = (input_A_matrix[edges_vert_1] - input_A_matrix[edges_vert_2]) ** 2

    return torch.sum(A_Frobe)


def body_shape_loss(smpl_vertices, real_points, vertex_normal, vertex_weight):
    x_nn = knn_points(smpl_vertices, real_points, lengths1=None, lengths2=None, K=1)
    dist = x_nn.dists[..., 0]
    idx = x_nn.idx[..., 0]

    u_vect = real_points[:, idx[0]] - smpl_vertices
    cosin_vect = torch.sum(u_vect * vertex_normal[:, idx[0]], dim=2)

    idx_out = cosin_vect > 0
    dist[idx_out] = vertex_weight * dist[idx_out]

    loss = torch.mean(dist)

    return loss


def body_shape_loss_inv(projected_vertices, real_points, body, k, device):
    x_nn = knn_points(projected_vertices, real_points, lengths1=None, lengths2=None, K=1)
    dist1 = x_nn.dists[..., 0]

    round_projected_vertices = torch.round(projected_vertices)
    y_nn = knn_points(round_projected_vertices, body, lengths1=None, lengths2=None, K=1)
    dist2 = y_nn.dists[..., 0]  # check projected points inside body or not
    id_out = (dist2 != 0.).nonzero()
    id_out = id_out[:, 1].clone()  # index of points outside body in dist1

    weight = torch.ones((dist1.shape[1], 1), device=device)
    weight[id_out] = k  # if point outside real body set the weight higher

    dist = weight.transpose(0, 1) * dist1

    loss = torch.mean(dist)

    return loss


def body_fitting_loss(body_pose, betas, model_joints, EXTCam, INTCam,
                      camera_x, camera_y, camera_z,
                      joints_2d, joints_conf, pose_prior,
                      projected_vertices=None, sigma=100, pose_prior_weight=4.78,
                      shape_prior_weight=5, angle_prior_weight=15.2,
                      output='sum', name='', vis=True):
    """
    Loss function for body fitting
    """
