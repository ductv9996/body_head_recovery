import cv2
import torch
import numpy as np
from .options.test_options import TestOptions
from .data.data_loader import CreateDataLoader
from .models.models import create_model
import torchvision.utils as utils
from body_head_recovery.Color_Transfer.util import util
from PIL import Image

from skimage import color
import torchvision.transforms as transforms


opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
model = create_model(opt)
opt.is_psnr = True

summary_dir = opt.results_dir
util.mkdirs([summary_dir])

def RGB2LAB(I):
    # AB 98.2330538631 -86.1830297444 94.4781222765 -107.857300207
    lab = color.rgb2lab(I)
    l = (lab[:, :, 0] / 100.0) #* 255.0    # L component ranges from 0 to 100
    a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) #* 255.0         # a component ranges from -127 to 127
    b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) #* 255.0         # b component ranges from -127 to 127
    #l = (lab[:, :, 0] / 100.0) * 255.0    # L component ranges from 0 to 100
    #a = (lab[:, :, 1] + 86.1830297444) / (98.2330538631 + 86.1830297444) * 255.0         # a component ranges from -127 to 127
    #b = (lab[:, :, 2] + 107.857300207) / (94.4781222765 + 107.857300207) * 255.0         # b component ranges from -127 to 127
    return np.dstack([l, a, b])


def get_transform_lab(): # Now we are using
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: RGB2LAB(np.array(img))))
    #transform_list.append(transforms.Lambda(lambda img: LAB2RGB(numpy.array(img))))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

transform_type = get_transform_lab()

def process_input(src_img_cv, tar_img_cv):
     
    # A_img = Image.open(A_path).convert('RGB')
    # B_img = Image.open(B_path).convert('RGB')
    A_img = Image.fromarray(cv2.cvtColor(src_img_cv, cv2.COLOR_BGR2RGB))
    B_img = Image.fromarray(cv2.cvtColor(tar_img_cv, cv2.COLOR_BGR2RGB))

    A = transform_type(A_img).unsqueeze(0)
    B = transform_type(B_img).unsqueeze(0)

    A_map = torch.zeros_like(A)
    B_map = torch.zeros_like(B)

    return {'A': A, 'B': B, 'A_map': A_map, 'B_map': B_map}


def run_transfer(src_img_cv, tar_img_cv):
    data = process_input(src_img_cv, tar_img_cv)
    model.set_input(data)
    model.test()

    result = model.out
    img_tensor =  util.tensor2im(result,model.img_type)
    grid = utils.make_grid(img_tensor, nrow=1, padding = 0, normalize = False)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    open_cv_image = np.array(im)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    open_cv_image = cv2.resize(open_cv_image, (2048, 2048), cv2.INTER_AREA)

    return open_cv_image



# src_img_path = "body_head_recovery/Color_Transfer/test/input/m_vang.png"
# tar_img_path = "body_head_recovery/Color_Transfer/test/target/final_texture.png"
# src_img_cv = cv2.imread(src_img_path)
# tar_img_cv = cv2.imread(tar_img_path)

# transfered_img = run_transfer(src_img_cv, tar_img_cv)

# cv2.imwrite("body_head_recovery/Color_Transfer/results/aa.png", open_cv_image)
