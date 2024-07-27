import torch

import torch.nn.functional as F
import torch.nn as nn

from utils import util


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

def tensor2image(tensor):
    # Detach the tensor from the computation graph and move it to CPU
    image = tensor.detach().cpu()
    image = (image * 255).clamp(0, 255)

    # Permute the dimensions to match the order (H, W, C) for image representation
    image = image.permute(1, 2, 0)

    # Convert from PyTorch tensor to uint8 data type
    image = image.to(torch.uint8)

    return image

class HEADPOST(nn.Module):
    def __init__(self, faces_expand, bary_coords, pix_to_face, uv_face_eye_mask):
        super().__init__()


        self.image_size = 224
        self.uv_size = 256

        self.faces_expand = faces_expand
        self.bary_coords = bary_coords
        self.pix_to_face = pix_to_face
        self.uv_face_eye_mask = uv_face_eye_mask


    def decompose_code(self, params):

        shape = params[:, 0:100]
        tex = params[:, 100:150]
        exp = params[:, 150:200]
        pose = params[:, 200:206]
        cam = params[:, 206:209]
        light = params[:, 209:236]

        return shape, tex, exp, pose, cam, light

    # # @torch.no_grad()
    def forward(self, image, cam, verts):

        trans_verts = util.batch_orth_proj(verts, cam)
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        
        uv_pverts = world2uv(trans_verts, self.pix_to_face, self.bary_coords, self.faces_expand)
        uv_gt = F.grid_sample(image, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear', align_corners=False)
        uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7)

        # uv_texture_gt = uv_gt[:, :3, :, :]
        # Detach the tensor from the computation graph and move it to CPU
        image_texture = uv_texture_gt.squeeze(0).detach().cpu()
        image_texture = (image_texture * 255).clamp(0, 255)

        # Permute the dimensions to match the order (H, W, C) for image representation
        image_texture = image_texture.permute(1, 2, 0)

        # Convert from PyTorch tensor to uint8 data type
        image_texture = image_texture.to(torch.uint8)

        return image_texture
    
    
    def process_image(self, image, image_landmarks):
        crop_size = 224
        scale = 1.25
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape

        bbox = self.detector(landmark=image_landmarks)
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
        old_size, center = self.bbox2point(left, right, top, bottom)
        size = int(old_size * scale)
        top = (center[1] - size / 2).to(dtype=torch.int)
        bot = (center[1] + size / 2).to(dtype=torch.int)
        left = (center[0] - size / 2).to(dtype=torch.int)
        right = (center[0] + size / 2).to(dtype=torch.int)
        image = image[top: bot, left:right]
        image = image.permute(2, 0, 1).unsqueeze(0)
        resized_image_tensor = F.interpolate(image, size=(crop_size, crop_size), mode='bilinear', align_corners=False)

        dst_image = resized_image_tensor / 255.

        return dst_image

    def detector(self, landmark):

        left = torch.min(landmark[:, 0])
        right = torch.max(landmark[:, 0])
        top = torch.min(landmark[:, 1])
        bottom = torch.max(landmark[:, 1])
        bbox = torch.stack([left, top, right, bottom], dim=0)

        return bbox

    def bbox2point(self, left, right, top, bottom):
        '''
        bbox from detector and landmarks are different
        '''

        old_size = (right - left + bottom - top) / 2 * 1.1
        center = torch.stack([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0], dim=0)

        return old_size, center
