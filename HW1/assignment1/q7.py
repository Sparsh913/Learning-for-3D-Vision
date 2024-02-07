import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import pytorch3d.io
from starter.utils import *
from pytorch3d.io import load_obj
from pytorch3d.vis.plotly_vis import plot_scene
from starter.render_mesh import render_cow
import imageio
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, get_points_renderer

def mesh2pcd(mesh, num_samples = None):
    '''
    write a function that takes a triangle mesh and the number of samples and outputs a 
    point cloud.
    1. Sample a face of a triangle in the mesh with probability proportional to the area 
    of the face
    2. Sample a random barycentric coordinate uniformly
    3. Compute the corresponding point using baricentric coordinates on the selected face.
    '''
    
    renderer = get_points_renderer(
        image_size=256, background_color=(1,1,1)
    )
    
    # Get the vertices and faces
    vertices, faces = mesh
    # vertices = vertices.unsqueeze(0)
    # faces = faces.unsqueeze(0)
    vertices = vertices.to(get_device())
    faces = faces.to(get_device())
    print("vertices shape: ", vertices.shape)
    print("faces shape: ", faces.shape)
    
    # Convert the vertices and faces to numpy
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    
    # Get the areas of the faces
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    f_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    f_areas = f_areas / np.sum(f_areas)
    
    # Sample n points. Let's first oversample and then we'll remove redundant points
    n_samples_per_face = np.ceil(num_samples * f_areas).astype(int)
    floor_n = np.sum(n_samples_per_face) - num_samples
    if floor_n > 0:
        idx = np.where(n_samples_per_face > 0)[0]
        floor_idx = np.random.choice(idx, floor_n, replace=True)
        n_samples_per_face[floor_idx] -= 1
        
    n_samples = np.sum(n_samples_per_face)
    
    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples, ), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample
        
    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
        np.sqrt(r[:,0:1]) * r[:,1:] * C
        
    print("P shape: ", P.shape)
    # Convert the numpy array to a pytorch tensor
    P = torch.tensor(P, dtype=torch.float32)
    features = torch.ones_like(P)
    point_cloud = pytorch3d.structures.Pointclouds(points=P.unsqueeze(0), features=features.unsqueeze(0))
    point_cloud = point_cloud.to(get_device())
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 6, elev = 0, azim = np.linspace(-180, 180, 12, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = get_device())
    
    rend = renderer(point_cloud.extend(12), cameras=cameras)
    print("rend device: ", rend.device)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    imageio.mimsave(f"output/q7_{num_samples}.gif", images, fps=10)

if __name__ == "__main__":
    mesh = load_cow_mesh(path="data/cow.obj")
    mesh2pcd(mesh, num_samples = 10)