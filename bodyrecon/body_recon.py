import json
import numpy as np

from .config_body import *
import smplx
import body_head_recovery.config as config

from .measure_app import measure_mesh
from .optimize_body import BodyOptim

def body_from_image_params(gender, body_image_f, height_m, weight_kg):

    
    if gender == "male":
        _A = male_A
        _B = male_B
    else:
        _A = female_A
        _B = female_B
    height_cm = height_m * 100.0

    v_root = pow(weight_kg, 1.0/3.0)
    measurements = torch.tensor([[height_cm], [v_root]])
    betas_10 = torch.mm(_A, measurements).transpose(0,1)[0] + _B
    betas_10 = betas_10.unsqueeze(0).to(config.DEVICE)

    optim = BodyOptim(image=body_image_f,
            step_size=1e-2,
            num_iters=100,
            device=config.DEVICE,
            visual=False)

    model_smplx = smplx.create(model_path=config.MODEL_PATH,
                                model_type=config.MODEL_TYPE, 
                                gender=gender, 
                                num_betas=10,
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=False,
                                dtype=torch.float32,)
    model_smplx.to(device=config.DEVICE)

    global_orient=torch.zeros(1,3, device=config.DEVICE, dtype=torch.float32)
    jaw_pose_f = torch.zeros(1,3,device=config.DEVICE)
    expression_f = torch.zeros(1,10,device=config.DEVICE)
    leye_pose_f = torch.zeros(1,3,device=config.DEVICE)
    reye_pose_f = torch.zeros(1,3,device=config.DEVICE)
    body_pose = torch.zeros(1, 63, device=config.DEVICE, dtype=torch.float32)
    body_pose[0, 38] = -np.pi/18
    body_pose[0, 41] = np.pi/18
    body_pose[0, 47] = -np.pi/4.5
    body_pose[0, 50] = np.pi/4.5
    model_show = model_smplx(global_orient=global_orient, 
                             betas=betas_10,
                             body_pose=body_pose,
                             jaw_pose=jaw_pose_f,
                             expression=expression_f,
                             leye_pose=leye_pose_f,
                             reye_pose=reye_pose_f)
    verts_show = model_show.vertices.detach().clone()
    joints_pos = model_show.joints.detach().clone()

    joints_pos[:, :, 1] = joints_pos[:, :, 1] - torch.min(verts_show[:, :, 1])
    verts_show[:, :, 1] = verts_show[:, :, 1] - torch.min(verts_show[:, :, 1])
    
    
    faces = model_smplx.faces
    measurement_user = measure_mesh(faces=faces, np_verts=verts_show.cpu().numpy().squeeze())

    # json_measurement_user = json.dumps(str(measurement_user))
    measurement_user = {key: str(np.round(value*100 + np.random.rand(), 2)) for key, value in measurement_user.items()}
    return verts_show.cpu().squeeze(),joints_pos.cpu().squeeze(), measurement_user

