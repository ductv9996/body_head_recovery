import os
import json
import torch
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from body_head_recovery.utils.renderer import SRenderY

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open("body_head_recovery/mapping/lowest_head_idx.json", "r") as f:
    lowest_head_idx = json.load(f)
with open("body_head_recovery/mapping/lowest_head_smplx_idx.json", "r") as fx:
    lowest_head_smplx_idx = json.load(fx)
smplx2head_idx = torch.tensor(np.load("body_head_recovery/mapping/smplx_head_correspondences/SMPL-X__Head_vertex_ids.npy"), dtype=torch.long)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda')
MODEL_PATH = "body_head_recovery/models/"
MODEL_TYPE = "smplx"
NUM_BETAS = 300

idx_map_glb_obj = torch.load("body_head_recovery/mapping/idx_map_glb_obj.pt")

# # head texture params
# faces_expand = torch.load("body_head_recovery/data/body_params/face_expand.pt")
# bary_coords = torch.load("body_head_recovery/data/body_params/bary_coords.pt")
# pix_to_face = torch.load("body_head_recovery/data/body_params/pix_to_face.pt")
# uv_face_eye_mask = torch.load("body_head_recovery/data/body_params/uv_face_eye_mask.pt")

image_size = 512

innerwear_male = cv2.imread("body_head_recovery/data/texture/innerwear_male.png")
innerwear_mask_male = cv2.imread("body_head_recovery/data/texture/mask_innerwear_male.png")/255.

innerwear_female = cv2.imread("body_head_recovery/data/texture/innerwear_female.png")
innerwear_mask_female = cv2.imread("body_head_recovery/data/texture/mask_innerwear_female.png")/255.


full_face_mask = cv2.imread("body_head_recovery/data/body_params/mask_full_face.png")/255.

eye_mask = cv2.imread('body_head_recovery/data/body_params/mask_eye.png')
head_mask = cv2.imread("body_head_recovery/data/body_params/head_mask.png")/255.
uv_face_front = cv2.imread("body_head_recovery/data/body_params/mask_face_eye.png")/255.

uv_face_right = cv2.imread("body_head_recovery/data/body_params/mask_face_right.png")/255.
uv_face_right_mask = uv_face_right[:,:, 0:1]
uv_face_right_mask = torch.tensor(uv_face_right_mask, dtype=torch.float32, device=DEVICE).permute(2, 0, 1).unsqueeze(0)

uv_face_left = cv2.imread("body_head_recovery/data/body_params/mask_face_left.png")/255.
uv_face_left_mask = uv_face_left[:,:, 0:1]
uv_face_left_mask = torch.tensor(uv_face_left_mask, dtype=torch.float32, device=DEVICE).permute(2, 0, 1).unsqueeze(0)

with open("body_head_recovery/mapping/right_mp_idx.json", "r") as fml:
    right_mp_idx = json.load(fml)
with open("body_head_recovery/mapping/left_mp_idx.json", "r") as fml:
    left_mp_idx = json.load(fml)

right_mp_idx = torch.tensor(right_mp_idx, dtype=torch.long)
left_mp_idx = torch.tensor(left_mp_idx, dtype=torch.long)

idx_smplx2mp = torch.load("body_head_recovery/mapping/smplx2mp_refine.pt").to(dtype=torch.long)

right_smplx_idx = idx_smplx2mp[right_mp_idx]
left_smplx_idx = idx_smplx2mp[left_mp_idx]

# idx for merger body and hair
body_head_idx = [8949, 8966, 1312, 447]
hair_head_idx = [3834, 3463, 1010, 4089]


idx_lmk_68_right = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 27, 28, 29, 30,
                    31, 32, 33, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 60, 61, 62,
                    59, 67, 66, 58, 57]
idx_lmk_68_left = [16, 15, 14, 13, 12, 11, 10, 9, 8, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                    33, 34, 35, 54, 53, 52, 51, 64, 63, 62, 55, 65, 66, 56, 57, 42, 43, 
                    44, 45, 46, 47]

mediapipe_landmark_embedding__smplx = np.load("body_head_recovery/mapping/mediapipe_landmark_embedding__smplx.npz")
idx_embedding_mp = mediapipe_landmark_embedding__smplx["landmark_indices"]
idx_embedding_smplx = mediapipe_landmark_embedding__smplx["lmk_face_idx"]


leftEyeUpper0= [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0= [263, 249, 390, 373, 374, 380, 381, 382, 362]

rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]

leftEyeIris = [473, 474, 475, 476, 477]
rightEyeIris = [468, 469, 470, 471, 472]

leftEyeUpper0_s = [1060, 1216, 1218, 1344, 9307, 1292, 1293]
leftEyeLower0_s= [1146, 1108, 815, 958, 883, 882, 827, 1361, 9287]

rightEyeUpper0_s = [2389, 2451, 2453, 2495, 9141, 2471, 2462]
rightEyeLower0_s = [2442, 2405, 2268, 2359, 2295, 2355, 2276, 2510, 9090]

rightEyeIris_s = [10049, 9996, 9963, 9932, 10027]
leftEyeIris_s = [9503, 9450, 9417, 9386, 9481]

nose = [168, 6, 197, 195, 5, 4]
nose_s = [8999, 9004, 9005, 8965, 8952, 9007]

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [95, 88, 178, 87, 14, 317, 402, 318, 324]

lipsUpperOuter_s = [2827, 2834, 2864, 2812, 2775, 8985, 1658, 1695, 1749, 1717, 1710]
lipsLowerOuter_s = [2880, 2898, 2905, 2948, 8947, 1865, 1802, 1795, 1773]
lipsUpperInner_s = [2845, 2848, 2852, 2787, 2786, 8975, 1669, 1670, 1737, 1733, 1711]
lipsLowerInner_s = [2894, 2896, 2907, 2938, 8948, 1849, 1804, 1793, 1791]


eye_brown = cv2.imread("body_head_recovery/data/texture/BlackBrown.png")
eye_blue = cv2.imread("body_head_recovery/data/texture/Green.png")


# print(idx_smplx2mp[lipsLowerOuter])

# idx_smplx2mp[leftEyeUpper0] = torch.tensor(leftEyeUpper0_s, dtype=torch.long)
# idx_smplx2mp[leftEyeLower0] = torch.tensor(leftEyeLower0_s, dtype=torch.long)
# idx_smplx2mp[rightEyeUpper0] = torch.tensor(rightEyeUpper0_s, dtype=torch.long)
# idx_smplx2mp[rightEyeLower0] = torch.tensor(rightEyeLower0_s, dtype=torch.long)
# idx_smplx2mp[leftEyeIris] = torch.tensor(leftEyeIris_s, dtype=torch.long)
# idx_smplx2mp[rightEyeIris] = torch.tensor(rightEyeIris_s, dtype=torch.long)
# idx_smplx2mp[nose] = torch.tensor(nose_s, dtype=torch.long)

# idx_smplx2mp[lipsUpperOuter] = torch.tensor(lipsUpperOuter_s, dtype=torch.long)
# idx_smplx2mp[lipsLowerOuter] = torch.tensor(lipsLowerOuter_s, dtype=torch.long)
# idx_smplx2mp[lipsUpperInner] = torch.tensor(lipsUpperInner_s, dtype=torch.long)
# idx_smplx2mp[lipsLowerInner] = torch.tensor(lipsLowerInner_s, dtype=torch.long)

# torch.save(idx_smplx2mp, "mapping/smplx2mp_refine.pt")


# load model for face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.2)

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='body_head_recovery/models/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1, min_face_detection_confidence=0.2)
face_detector = vision.FaceLandmarker.create_from_options(options)

render = SRenderY(image_size, obj_filename="body_head_recovery/data/smplx_uv/smplx_uv.obj", uv_size=2048).to(DEVICE)


idx_face = [2433, 2595, 2497, 2514, 2521, 2540, 3809, 1348, 1014, 1164, 979, 908, 2819, 2812, 3541, 1695, 1702,
                1801, 3511, 2899]
idx_head = [36613, 61856, 60859, 58126, 58040, 57920, 67455, 65645, 67786, 18480, 18379, 2957, 57318, 56751, 56880,
               63151, 62408, 63950, 59127, 59071]

