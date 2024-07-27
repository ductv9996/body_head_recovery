import cv2
import torch
import smplx

import body_head_recovery.config as config
from body_head_recovery.utils.uv_utils import batch_orth_proj

class HEAD_OPTIM:
    def __init__(self, gender, vis):

        self.model_smplx = smplx.create(model_path=config.MODEL_PATH,
                                    model_type=config.MODEL_TYPE, 
                                    gender=gender, 
                                    num_betas=config.NUM_BETAS,
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
        self.device = config.DEVICE
        self.model_smplx.to(device=self.device)
        self.vis = vis

    def __call__(self, image_f, landmarks_f, image_r, landmarks_r, image_l, landmarks_l):

        betas = torch.zeros(1, config.NUM_BETAS, device=self.device, dtype=torch.float32)
        global_orient_f = torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        camera_f = torch.tensor([[5.6529,  0.0382, -0.2883]],device=self.device)

        global_orient_r = torch.tensor([[-0.0126,  0.65,  0.0028]], device=self.device, dtype=torch.float32)
        camera_r = torch.tensor([[5.9164e+00, -4.5812e-03, -2.9479e-01]],device=self.device)

        global_orient_l = torch.tensor([[-0.0231, -0.65,  0.1219]], device=self.device, dtype=torch.float32)
        camera_l = torch.tensor([[5.7402,  0.0715, -0.2840]],device=self.device)

        jaw_pose_f = torch.zeros(1,3,device=self.device)
        expression_f = torch.zeros(1,10,device=self.device)
        leye_pose_f = torch.zeros(1,3,device=self.device)
        reye_pose_f = torch.zeros(1,3,device=self.device)

        jaw_pose_r = torch.zeros(1,3,device=self.device)
        expression_r = torch.zeros(1,10,device=self.device)
        leye_pose_r = torch.zeros(1,3,device=self.device)
        reye_pose_r = torch.zeros(1,3,device=self.device)

        jaw_pose_l = torch.zeros(1,3,device=self.device)
        expression_l = torch.zeros(1,10,device=self.device)
        leye_pose_l = torch.zeros(1,3,device=self.device)
        reye_pose_l = torch.zeros(1,3,device=self.device)

        img_lmks_f = landmarks_f.detach().clone().to(device=self.device)
        img_lmks_f = img_lmks_f.unsqueeze(0)

        img_lmks_r = landmarks_r.detach().clone().to(device=self.device)
        img_lmks_r = img_lmks_r.unsqueeze(0)

        img_lmks_l = landmarks_l.detach().clone().to(device=self.device)
        img_lmks_l = img_lmks_l.unsqueeze(0)


        # Step 1: Optimize pose, betas, camera
        betas.requires_grad = False
        global_orient_f.requires_grad = True
        camera_f.requires_grad = True
        expression_f.requires_grad = False
        jaw_pose_f.requires_grad=False
        leye_pose_f.requires_grad=False
        reye_pose_f.requires_grad=False
        
        global_orient_r.requires_grad = True
        camera_r.requires_grad = True
        expression_r.requires_grad = False
        jaw_pose_r.requires_grad=False
        leye_pose_r.requires_grad=False
        reye_pose_r.requires_grad=False

        global_orient_l.requires_grad = True
        camera_l.requires_grad = True
        expression_l.requires_grad = False
        jaw_pose_l.requires_grad=False
        leye_pose_l.requires_grad=False
        reye_pose_l.requires_grad=False
        
        shape_opt_params = [global_orient_f, camera_f, 
                            global_orient_r, camera_r,
                            global_orient_l, camera_l, ]
        # shape_opt_params = [camera]
        shape_optimizer = torch.optim.Adam(shape_opt_params,
                                            lr=1e-2,
                                            betas=(0.9, 0.999))
        for i in range(300):

            model_output_f = self.model_smplx(global_orient=global_orient_f, betas=betas,
                                        jaw_pose=jaw_pose_f,
                                        expression=expression_f,
                                        leye_pose=leye_pose_f,
                                        reye_pose=reye_pose_f)
            
            model_output_r = self.model_smplx(global_orient=global_orient_r, betas=betas,
                                        jaw_pose=jaw_pose_r,
                                        expression=expression_r,
                                        leye_pose=leye_pose_r,
                                        reye_pose=reye_pose_r)
            
            model_output_l = self.model_smplx(global_orient=global_orient_l, betas=betas,
                                        jaw_pose=jaw_pose_l,
                                        expression=expression_l,
                                        leye_pose=leye_pose_l,
                                        reye_pose=reye_pose_l)
            
            dist_f, proj_lmk_2D_f, model_verts_f = loss_lmks(model_output=model_output_f, map_idx=config.idx_smplx2mp, cam=camera_f, lmks_2D=img_lmks_f)
            dist_r, proj_lmk_2D_r, model_verts_r = loss_lmks(model_output=model_output_r, map_idx=config.idx_smplx2mp, cam=camera_r, lmks_2D=img_lmks_r)
            dist_l, proj_lmk_2D_l, model_verts_l = loss_lmks(model_output=model_output_l, map_idx=config.idx_smplx2mp, cam=camera_l, lmks_2D=img_lmks_l)

            loss = torch.mean(dist_f) + torch.mean(dist_r) + torch.mean(dist_l)

            if self.vis:
                img_sum = cv2.hconcat([image_f, image_r, image_l]) 
                
                for p in img_lmks_f[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_f[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]), int(p1[1])), 2, (0, 0, 255), -1)

                for p in img_lmks_r[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]) + 512, int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_r[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]) + 512, int(p1[1])), 2, (0, 0, 255), -1)

                for p in img_lmks_l[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]) + 512*2, int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_l[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]) + 512*2, int(p1[1])), 2, (0, 0, 255), -1)

                cv2.imshow("bb", img_sum)
                cv2.waitKey(1)
            
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()

        # Step 2: Optimize pose, betas, camera
        betas.requires_grad = True
        global_orient_f.requires_grad = True
        camera_f.requires_grad = True
        expression_f.requires_grad = True
        jaw_pose_f.requires_grad=True
        leye_pose_f.requires_grad=True
        reye_pose_f.requires_grad=True
        
        global_orient_r.requires_grad = True
        camera_r.requires_grad = True
        expression_r.requires_grad = True
        jaw_pose_r.requires_grad=True
        leye_pose_r.requires_grad=True
        reye_pose_r.requires_grad=True

        global_orient_l.requires_grad = True
        camera_l.requires_grad = True
        expression_l.requires_grad = True
        jaw_pose_l.requires_grad=True
        leye_pose_l.requires_grad=True
        reye_pose_l.requires_grad=True
        
        shape_opt_params = [betas, global_orient_f, camera_f, expression_f, jaw_pose_f, leye_pose_f, reye_pose_f, 
                            global_orient_r, camera_r, expression_r, jaw_pose_r, leye_pose_r, reye_pose_r, 
                            global_orient_l, camera_l, expression_l, jaw_pose_l, leye_pose_l, reye_pose_l, ]
        # shape_opt_params = [camera]
        shape_optimizer = torch.optim.Adam(shape_opt_params,
                                            lr=1e-3,
                                            betas=(0.9, 0.999))
        for i in range(800):

            model_output_f = self.model_smplx(global_orient=global_orient_f, betas=betas,
                                        jaw_pose=jaw_pose_f,
                                        expression=expression_f,
                                        leye_pose=leye_pose_f,
                                        reye_pose=reye_pose_f)
            
            model_output_r = self.model_smplx(global_orient=global_orient_r, betas=betas,
                                        jaw_pose=jaw_pose_r,
                                        expression=expression_r,
                                        leye_pose=leye_pose_r,
                                        reye_pose=reye_pose_r)
            
            model_output_l = self.model_smplx(global_orient=global_orient_l, betas=betas,
                                        jaw_pose=jaw_pose_l,
                                        expression=expression_l,
                                        leye_pose=leye_pose_l,
                                        reye_pose=reye_pose_l)
            
            dist_f, proj_lmk_2D_f, model_verts_f = loss_lmks(model_output=model_output_f, map_idx=config.idx_smplx2mp, cam=camera_f, lmks_2D=img_lmks_f)
            dist_r, proj_lmk_2D_r, model_verts_r = loss_lmks(model_output=model_output_r, map_idx=config.right_smplx_idx, cam=camera_r, lmks_2D=img_lmks_r[:, config.right_mp_idx])
            dist_l, proj_lmk_2D_l, model_verts_l = loss_lmks(model_output=model_output_l, map_idx=config.left_smplx_idx, cam=camera_l, lmks_2D=img_lmks_l[:, config.left_mp_idx])

            dist_f[:, config.idx_embedding_mp] = 10*dist_f[:, config.idx_embedding_mp]
            loss = 2*torch.mean(dist_f) + torch.mean(dist_r) + torch.mean(dist_l)

            if self.vis:
                img_sum = cv2.hconcat([image_f, image_r, image_l]) 

                for p in img_lmks_f[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_f[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]), int(p1[1])), 2, (0, 0, 255), -1)

                for p in img_lmks_r[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]) + 512, int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_r[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]) + 512, int(p1[1])), 2, (0, 0, 255), -1)

                for p in img_lmks_l[0].cpu().numpy():
                    cv2.circle(img_sum, (int(p[0]) + 512*2, int(p[1])), 2, (255, 0, 0), -1)
                for p1 in proj_lmk_2D_l[0].detach().clone().cpu().numpy():
                    cv2.circle(img_sum, (int(p1[0]) + 512*2, int(p1[1])), 2, (0, 0, 255), -1)

                cv2.imshow("bb", img_sum)
                cv2.waitKey(1)
            
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()
        
        if self.vis:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # print("global_orient: ", global_orient)
        # print("camera: ", camera)

        model_verts_f = model_verts_f.detach().clone()
        camera_f = camera_f.detach().clone()

        model_verts_r = model_verts_r.detach().clone()
        camera_r = camera_r.detach().clone()

        model_verts_l = model_verts_l.detach().clone()
        camera_l = camera_l.detach().clone()

        model_show = self.model_smplx(global_orient=torch.zeros(1,3, device=self.device, dtype=torch.float32), betas=betas,
                                        jaw_pose=jaw_pose_f,
                                        expression=expression_f,
                                        leye_pose=leye_pose_f,
                                        reye_pose=reye_pose_f)
        verts_show = model_show.vertices.detach().clone()
        verts_show[:, :, 1] = verts_show[:, :, 1] - torch.min(verts_show[:, :, 1])

        return verts_show, model_verts_f, camera_f, model_verts_r, camera_r, model_verts_l, camera_l
    

def loss_lmks(model_output, map_idx, cam, lmks_2D):

    model_verts = model_output.vertices
    smplx_lmks = model_verts[:, map_idx]
    proj_landmarks2d = batch_orth_proj(smplx_lmks, cam)
    proj_landmarks2d[:, :, 1:] = -proj_landmarks2d[:, :,1:]
    proj_landmarks2d = proj_landmarks2d[:, :, :2] * config.image_size/2 + config.image_size/2 -1 
    # loss = torch.mean((proj_landmarks2d[:, :, :2] - lmks_2D)**2)
    dist = (proj_landmarks2d[:, :, :2] - lmks_2D)**2

    return dist, proj_landmarks2d, model_verts