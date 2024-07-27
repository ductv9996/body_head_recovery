import torch
import torch.nn.functional as F


def process_image(image, image_landmarks):
    crop_size = 512
    # scale = 1.25
    scale = 3.
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
    resized_image_tensor = F.interpolate(image, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
    dst_image = resized_image_tensor / 255.

    process_landmarks = image_landmarks - torch.stack([left, top])
    process_landmarks = process_landmarks*crop_size/image.shape[-1]

    return dst_image, process_landmarks

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