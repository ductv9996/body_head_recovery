import torch
import torch.nn.functional as F
import numpy as np
import cv2

import body_head_recovery.config as config


def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    # assert (vertices.ndimension() == 3)
    # assert (faces.ndimension() == 3)
    # assert (vertices.shape[0] == faces.shape[0])
    # assert (vertices.shape[2] == 3)
    # assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def world2uv(vertices, pix_to_face, bary_coords, faces_expand):
    '''
    warp vertices from world space to uv space
    vertices: [bz, V, 3]
    uv_vertices: [bz, 3, h, w]
    '''
    attributes = face_vertices(vertices, faces_expand)

    vismask = (pix_to_face > -1).float()
 
    D = attributes.shape[-1]
    attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
    N, H, W, K, _ = bary_coords.shape
    mask = pix_to_face == -1
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
    pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)

    uv_vertices = pixel_vals[:, :3]
    
    return uv_vertices


def batch_orth_proj(verts, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    src = verts.clone()
    camera = camera.clone().view(-1, 1, 3)
    X_trans = src[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, src[:, :, 2:]], 2)

    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


# -------------------------------------- io
def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                # print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            # print('copy param {} failed'.format(k))
            continue


def crop_image(image, image_landmarks):
    """ image_landmarks: tensor nx2 """
    crop_size = config.image_size
    scale = 2.

    h, w, _ = image.shape

    bbox = detector(landmark=image_landmarks)
    
    left = bbox[0]
    right = bbox[2]
    top = bbox[1]
    bottom = bbox[3]
    old_size, center = bbox2point(left, right, top, bottom)
    size = int(old_size * scale)
    top = int(center[1] - size / 2)
    bot = int(center[1] + size / 2)
    left = int(center[0] - size / 2)
    right = int(center[0] + size / 2)
    
    b_top=0
    b_left=0
    b_bot=0
    b_right = 0
    if top < 0:
        b_top = abs(top)
        top = 0
    if left < 0:
        b_left = abs(left)
        left = 0
    if bot > h-1:
        b_bot = abs(bot - h+1)
        bot = h-1
    if right > w-1:
        b_right = abs(right - w+1)
        right = w-1

    image = image[top: bot, left:right]
    image_landmarks = image_landmarks - torch.tensor([left-b_left, top-b_top])
    image = cv2.copyMakeBorder(image, b_top, b_bot, b_left, b_right, cv2.BORDER_CONSTANT, None, value = 0)
    
    
    image_landmarks = image_landmarks*crop_size/image.shape[0]
    image = cv2.resize(image, (crop_size, crop_size), cv2.INTER_AREA)
    return image, image_landmarks



def process_image(image, image_landmarks):
    # crop_size = 224
    # scale = 1.25
    crop_size = 512
    scale = 2.

    if len(image.shape) == 2:
        image = image[:, :, None].repeat(1, 1, 3)
    if len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    h, w, _ = image.shape

    bbox = detector(landmark=image_landmarks)
    if bbox.shape[0] < 4:
        left = torch.tensor(0)
        right = torch.tensor(h - 1)
        top = torch.tensor(0)
        bottom = torch.tensor(w - 1)
    else:
        left = bbox[0]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]
    old_size, center = bbox2point(left, right, top, bottom)
    size = int(old_size * scale)
    top = (center[1] - size / 2).to(dtype=torch.int)
    bot = (center[1] + size / 2).to(dtype=torch.int)
    left = (center[0] - size / 2).to(dtype=torch.int)
    right = (center[0] + size / 2).to(dtype=torch.int)
    image = image[top: bot, left:right]
    image = image.permute(2, 0, 1).unsqueeze(0)

    image_landmarks = image_landmarks - torch.stack([left, top])
    image_landmarks = image_landmarks*crop_size/image.shape[-1]
    resized_image_tensor = F.interpolate(image, size=(crop_size, crop_size), mode='bilinear', align_corners=False)

    dst_image = resized_image_tensor / 255.

    return dst_image, image_landmarks

def detector(landmark):

    left = torch.min(landmark[:, 0])
    right = torch.max(landmark[:, 0])
    top = torch.min(landmark[:, 1])
    bottom = torch.max(landmark[:, 1])
    bbox = torch.stack([left, top, right, bottom], dim=0)

    return bbox

def bbox2point(left, right, top, bottom):
    '''
    bbox from detector and landmarks are different
    '''

    old_size = (right - left + bottom - top) / 2 * 1.1
    center = torch.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=0)

    return old_size, center

def tensor2image(tensor):
    # Detach the tensor from the computation graph and move it to CPU
    image = tensor.detach().cpu()
    image = (image * 255).clamp(0, 255)

    # Permute the dimensions to match the order (H, W, C) for image representation
    image = image.permute(1, 2, 0)

    # Convert from PyTorch tensor to uint8 data type
    image = image.to(torch.uint8)

    return image


def compute_similarity_transform_torch(S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T

        assert (S2.shape[1] == S1.shape[1])

        mu1= torch.mean(S1, dim=1, keepdim=True).to(dtype=torch.float32)
        mu2= torch.mean(S2, dim=1, keepdim=True).to(dtype=torch.float32)
        X1 = S1 - mu1
        X2 = S2 - mu2

        var1 = torch.sum(X1 ** 2).to(dtype=torch.float32)

        K = X1.mm(X2.T).to(torch.float32)

        U, s, V = torch.svd(K)
        # V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[0], device=S1.device, dtype=torch.float32)
        Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
        # Construct R.
        
        V = V.to(dtype=torch.float32)
        U = U.to(dtype=torch.float32)       
        R = V.mm(Z.mm(U.T))

        # print('R', X1.shape)

        scale = torch.trace(R.mm(K)) / var1
        # print(R.shape, mu1.shape)
        
        t = mu2 - scale * (R.mm(mu1))

        return scale, R, t


def transfer_texture(body_verts, cam, croped_img):

    trans_verts = batch_orth_proj(body_verts, cam)
    trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

    image_tensor = torch.tensor(croped_img, dtype=torch.float32, device=config.DEVICE)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor / 255.

    uv_pverts = config.render.world2uv(trans_verts.to(config.DEVICE))
    uv_gt = F.grid_sample(image_tensor, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear')
    # uv_texture_gt = uv_gt[:, :3, :, :] * config.uv_face_eye_mask + (
    #             torch.ones_like(uv_gt[:, :3, :, :]) * (1 - config.uv_face_eye_mask) * 0.7)
    uv_texture_gt = uv_gt[:, :3, :, :]

    # Detach the tensor from the computation graph and move it to CPU
    image_texture = uv_texture_gt.squeeze(0).detach().cpu()
    image_texture = (image_texture * 255).clamp(0, 255)

    # Permute the dimensions to match the order (H, W, C) for image representation
    image_texture = image_texture.permute(1, 2, 0)

    # Convert from PyTorch tensor to uint8 data type
    image_texture = image_texture.to(torch.uint8)

    return image_texture.numpy()
