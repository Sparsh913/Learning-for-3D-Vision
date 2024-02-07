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
    
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces])
    face_areas = mesh.faces_areas_packed()
    face_areas = face_areas / face_areas.sum()
    areas = face_areas.cpu().numpy()
    
    
    # Sample a face of a triangle in the mesh with probability proportional to the area
    # face_indices = torch.multinomial(face_areas, num_samples, replacement=True)
    face_indices = np.random.choice(len(areas), num_samples, p=areas)
    print("face_indices shape: ", face_indices.shape)
    
    # Sample a random barycentric coordinate uniformly
    u = torch.rand(num_samples, 1).to(get_device())
    
    # Find the corresponding point using barycentric coordinates on the selected face
    # Get the vertices of the selected faces
    v0 = vertices[faces[face_indices, 0],]
    v1 = vertices[faces[face_indices, 1],]
    v2 = vertices[faces[face_indices, 2],]
    print("v0 shape: ", v0.shape)
    
    # Number of points to sample on each face -> corresponding to the area of the face
    # n_points_per_face = torch.ceil(face_areas * num_samples).int()
    # print("n_points_per_face shape: ", n_points_per_face.shape)
    # print("total points: ", n_points_per_face.sum())
    
    # Sample the points using the barycentric coordinates
    # For two random variables r1,r2 uniformly distributed from 0 to 1, 
    # we sample a new point d
    # d = (1 - sqrt(r1)) * v0 + sqrt(r1) * (1 - r2) * v1 + sqrt(r1) * r2 * v2
    r1 = torch.rand(num_samples, 1).to(get_device())
    r2 = torch.rand(num_samples, 1).to(get_device())
    P = (1 - torch.sqrt(r1)) * v0 + torch.sqrt(r1) * (1 - r2) * v1 + torch.sqrt(r1) * r2 * v2
    # P = v0 *r1 + v1 * r2 + v2 * (1 - r1 - r2)
    print("P shape: ", P.shape)
    
    features = torch.ones_like(P)
    point_cloud = pytorch3d.structures.Pointclouds(points=P.unsqueeze(0), features=features.unsqueeze(0))
    point_cloud = point_cloud.to(get_device())
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 2, elev = 0, azim = np.linspace(-180, 180, 12, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = get_device())
    
    rend = renderer(point_cloud.extend(12), cameras=cameras)
    print("rend device: ", rend.device)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    imageio.mimsave(f"output/q7/{num_samples}.gif", images, fps=10, loop=50)
    

if __name__ == "__main__":
    # Load the cow mesh
    cow_mesh = load_cow_mesh(path="data/cow.obj")
    # Convert the cow mesh to a point cloud
    mesh2pcd(cow_mesh, num_samples=10000)