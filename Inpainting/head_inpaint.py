import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from .src.model.aotgan import InpaintGenerator
from body_head_recovery.Inpainting import config_inpaint

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

model = InpaintGenerator()
model.load_state_dict(torch.load(config_inpaint.pre_train, map_location="cuda"))
model.eval()

def run_inpaint(orig_img):

    crop_img = orig_img[:1220, :1220]

    resize_img = cv2.resize(crop_img, (512, 512))
    img_tensor = (ToTensor()(resize_img) * 2.0 - 1.0).unsqueeze(0)

    mask = cv2.imread("body_head_recovery/data/body_params/mask_inpaint.png", cv2.IMREAD_GRAYSCALE)
    
    with torch.no_grad():

        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        # comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

        pred_np = postprocess(pred_tensor[0])
        pred_np = cv2.resize(pred_np, (1220, 1220), cv2.INTER_AREA)

        pred_inpaint = orig_img.copy()
        pred_inpaint[:1220, :1220] = pred_np

    return pred_inpaint
