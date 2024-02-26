import argparse
import os
import time

import losses
from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import dataset_location
import torch
import pytorch3d
import mcubes
import numpy as np
from utils import get_mesh_renderer, get_points_renderer
import imageio
from render import mesh_to_image, points_to_image
from pytorch3d.io import load_obj


def get_args_parser():
    parser = argparse.ArgumentParser('Model Fit', add_help=False)
    parser.add_argument('--lr', default=4e-5, type=float)
    parser.add_argument('--max_iter', default=100000, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--device', default='cuda', type=str) 
    return parser

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)
    # Save the mesh_src to a file as obj
    verts, faces = mesh_src.verts_packed(), mesh_src.faces_packed()
    # faces = faces.verts_idx
    pytorch3d.io.save_obj('outputs/q1_3_src.obj', verts, faces)

    print('Done!')
    
    return mesh_src


def fit_pointcloud(pointclouds_src, pointclouds_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([pointclouds_src], lr = args.lr)
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()
        print("pointclouds_src shape: ", pointclouds_src.shape)
        print("pointclouds_tgt shape: ", pointclouds_tgt.shape)
        loss = losses.chamfer_loss(pointclouds_src, pointclouds_tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    # Save the pointclouds_src to a file as npz
    # np.save('outputs/q1_2_src.npy', pointclouds_src.squeeze(0).cpu().detach().numpy())
    print('Done!')
    return pointclouds_src


def fit_voxel(voxels_src, voxels_tgt, args):
    start_iter = 0
    start_time = time.time()    
    optimizer = torch.optim.Adam([voxels_src], lr = args.lr)
    
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        loss = losses.voxel_loss(voxels_src,voxels_tgt)
        # print("loss_voxel shape: ", loss.shape)
        # print("loss_voxel device: ", loss.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))
    
    print('Done!')
    
    return voxels_src


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)

    
    feed = r2n2_dataset[0]


    feed_cuda = {}
    for k in feed:
        if torch.is_tensor(feed[k]):
            feed_cuda[k] = feed[k].to(args.device).float()


    if args.type == "vox":
        # initialization
        voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
        # print("voxels_src shape: ", voxels_src.shape)
        voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
        voxels_tgt = feed_cuda['voxels']
        # print("voxels_tgt shape: ", voxels_tgt.shape)

        # fitting
        voxel_src = fit_voxel(voxels_src, voxels_tgt, args)
        # Visualize the voxel grid alongside the target voxel grid
        # visualize_voxel_grid(voxel_src)
        # mesh_1 = pytorch3d.ops.cubify(voxels = voxels_src, thresh = 0.5)
        # mesh_2 = pytorch3d.ops.cubify(voxels_tgt, thresh = 0.5, device = args.device)
        # print("mesh_1 type: ", type(mesh_1))
        
        # Visualize voxel grid as a mesh using marching cubes
        # shift voxel_src to cpu and then convert to numpy
        voxel_src = voxel_src.cpu()
        voxel_src = voxel_src.detach().numpy()
        mesh = mcubes.marching_cubes(voxel_src[0], 0.5)
        
        voxels_tgt = voxels_tgt.cpu()
        voxels_tgt = voxels_tgt.detach().numpy()
        mesh_tgt = mcubes.marching_cubes(voxels_tgt[0], 0.5)
        
        
        mesh_to_image(mesh, out_path = 'outputs/q1_1_src', dist = 68, elev = 18, at = ((0, 8, 7),))
        mesh_to_image(mesh_tgt, out_path = 'outputs/q1_1_tgt', dist = 68, elev = 18, at = ((0, 8, 7),))
        # # convert to torch tensor
        # verts1 = torch.tensor(verts1, device=args.device)
        # # convert faces1 to numpy float32
        # faces1 = faces1.astype(np.float32)
        # faces1 = torch.tensor(faces1, device=args.device, dtype=torch.float32)
        # verts1 = verts1.unsqueeze(0)
        # faces1 = faces1.unsqueeze(0)
        # renderer = get_mesh_renderer()
        # num_views = 10
        # mesh1 = pytorch3d.structures.Meshes(verts1, faces1).to(args.device)
        # # Prepare the camera
        # R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=10, azim=np.linspace(-180, 180, 10, endpoint=False), device=args.device)
        # many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = args.device, R=R, T=T)
        # lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=args.device)
        
        # images = renderer(mesh1.extend(num_views), cameras=many_cameras, lights=lights)
        # images = [image[:,:,:3] for image in images.cpu().numpy()]
        # images = [np.uint8(image*255) for image in images]
        # imageio.mimsave('output/q1_1.gif', images, fps=10, loop=5000) 


    elif args.type == "point":
        # initialization
        pointclouds_src = torch.randn([1,args.n_points,3],requires_grad=True, device=args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        pointclouds_tgt = sample_points_from_meshes(mesh_tgt, args.n_points)
        # print("pointclouds_tgt shape: ", pointclouds_tgt.shape) # (1, 5000, 3)
        # print(pointclouds_src.squeeze(0).shape)

        # fitting
        pcd = fit_pointcloud(pointclouds_src, pointclouds_tgt, args) # Run only once
        
        # # Visualize the point cloud
        # load npz file and visualize the point cloud
        pcd = np.load('outputs/q1_2_src.npy')
        # Convert to torch tensor
        pcd = torch.tensor(pcd, device=args.device, dtype=torch.float32)
        points_to_image(pcd, out_path = 'outputs/q1_2_src')
        points_to_image(pointclouds_tgt, out_path = 'outputs/q1_2_tgt')
        
             
    
    elif args.type == "mesh":
        # initialization
        # try different ways of initializing the source mesh        
        mesh_src = ico_sphere(4, args.device)
        mesh_tgt = Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
        # print("mesh_tgt: ", mesh_tgt)
        # print(mesh_tgt.verts_packed().shape)
        # print(mesh_tgt.faces_packed().shape)

        # fitting
        mesh = fit_mesh(mesh_src, mesh_tgt, args) # Run only once
        
        # Visualize the mesh
        mesh = pytorch3d.io.load_obj('outputs/q1_3_src.obj')
        # mesh_to_image(mesh, out_path = 'outputs/q1_3_src', dist=2, elev = 0)
        mesh_to_image(mesh_tgt, out_path = 'outputs/q1_3_tgt', dist = 2, elev = 0)
        # Render the mesh
        vertices, faces, _ = mesh
        faces = faces.verts_idx
        vertices = torch.tensor(vertices, dtype=torch.float32).to(args.device).unsqueeze(0)
        faces = torch.tensor(faces, dtype=torch.float32).to(args.device).unsqueeze(0)
        textures = torch.ones_like(vertices, device=args.device, dtype=torch.float32)
        color = torch.tensor([0.7, 0.7, 1], dtype=torch.float32).to(args.device)
        textures = textures * color
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(args.device)
        renderer = get_mesh_renderer()
        device = args.device
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=3, elev=0, azim=np.linspace(-180, 180, 12, endpoint=False), 
            at = ((0, 0, 0),))
        out_path = 'outputs/q1_3_src'
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
        images = renderer(mesh.extend(12), cameras=many_cameras, lights=lights) # images containing all the views
        # Convert to numpy and transpose the image dimensions
        images = [image[:,:,:3] for image in images.cpu().numpy()] 
        print("images shape: ", images[0].shape)
        # each image should be uint8
        images = [np.uint8(image*255) for image in images]
        # Convert this list contatining each view into a gif
        imageio.mimsave(out_path + '.gif', images, fps=10, loop=5000)
        print("Gif saved at: ", out_path + '.gif')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Fit', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
