import torch
import json
# import open3d as o3d
# from open3d import *
import numpy as np
from .losses import body_shape_loss_1
from body_head_recovery import config as config

class BodyOptim():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 image=None,
                 step_size=1e-2,
                 num_iters=100,
                 device=torch.device('cpu'),
                 visual=False):

        # Store options
        self.device = device
        self.step_size = step_size
        self.flame = "Flame_model"
        self.num_iters = num_iters

        self.idx_face = torch.tensor(config.idx_face, device=device, dtype=torch.long)
        self.idx_head = torch.tensor(config.idx_head, device=device, dtype=torch.long)

        # Load parametric model
        self.visual = visual
        # if self.visual:
        #     self.obj_pcd = geometry.PointCloud()
        #     self.obj_pcd_1 = geometry.PointCloud()
        #     self.obj_pcd_2 = geometry.PointCloud()
        #     self.vis = visualization.Visualizer()
        #     self.vis.create_window(window_name='STEP optim on point cloud', width=800, height=800)
        #     self.vis.get_render_option().point_size = 3

    def __call__(self, human_pcl, init_cam, init_shape, init_pose, init_exp):
        """optimize_pcl: Use normalized height 1m fit to the predicted model from heightweight2Model
        Perform body fitting.
            1 step:
        Input:
            gt_betas: model 10 predict from previous parts
            init_cam:  init camera position to render model
        Returns:
            betas: 10 parameters of normalized height 1m model
        """
        camera = init_cam.detach().clone()
        pose = init_pose.clone()
        shape = init_shape.clone()
        exp = init_exp.clone()

        pose.requires_grad = False
        camera.requires_grad = True
        shape.requires_grad = True
        exp.requires_grad = True
        shape_opt_params = [camera, shape, exp]
        shape_optimizer = torch.optim.Adam(shape_opt_params,
                                           lr=self.step_size,
                                           betas=(0.9, 0.999))

        first_loop = True
        for i in range(self.num_iters):
            verts, landmarks2d, landmarks3d = self.flame(shape_params=shape, expression_params=exp,
                                                         pose_params=pose)
            model_vert = verts + camera

            shape_loss = body_shape_loss_1(model_vertices=model_vert[:, self.idx_head], human_vertices=human_pcl)
            # shape_loss = body_shape_loss(smpl_vertices=model_vert[:, self.idx_head], real_points=human_pcl,
            #                              vertex_normal=self.vertex_normal, vertex_weight=5)
            # print('shape_loss: ', shape_loss, 'shape_loss_1: ', 0.1*shape_loss_1)

            loss = shape_loss

            # if self.visual:
            #     smpl_vert = model_vert.detach().clone().cpu().numpy().squeeze()
            #     human_vert = human_pcl.detach().clone().cpu().numpy().squeeze()
            #     self.obj_pcd.points = utility.Vector3dVector(smpl_vert)
            #     self.obj_pcd.paint_uniform_color([1, 0, 0])
            #     self.obj_pcd_1.points = utility.Vector3dVector(human_vert)
            #     self.obj_pcd_1.paint_uniform_color([0, 0, 1])
            #     if first_loop:
            #         self.vis.add_geometry(self.obj_pcd)
            #         self.vis.add_geometry(self.obj_pcd_1)
            #         first_loop = False
            #     else:
            #         self.vis.update_geometry(self.obj_pcd)
            #         self.vis.update_geometry(self.obj_pcd_1)
            #         self.vis.poll_events()
            #         self.vis.update_renderer()
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()

        # if self.visual:
        #     self.vis.destroy_window()
        #     o3d.visualization.draw_geometries([self.obj_pcd, self.obj_pcd_1])

        return model_vert.detach().clone().cpu().numpy().squeeze()