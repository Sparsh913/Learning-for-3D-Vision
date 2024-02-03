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

def q2_1(num_views=12):
    '''
    Construct a tetrahedron mesh and then render it from multiple viewpoints.
    Manually define the vertices and faces of the mesh.
    '''
    device = get_device()
    renderer = get_mesh_renderer(image_size=256)
    
    # Define the vertices and faces
    vertices = torch.tensor([[-1,0,0], [1,0,0], [0,1,0], [0,0.5,1]], dtype=torch.float32, device=device)
    faces = torch.tensor([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=torch.int64, device=device)
    faces = faces.unsqueeze(0)
    vertices = vertices.unsqueeze(0)
    print("vertices shape: ", vertices.shape)
    print("faces shape: ", faces.shape)
    
    # Adding a single-color texture
    texture_rgb = torch.ones_like(vertices) # N X 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1.0], device=device)
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb)
    print("texture rgb", texture_rgb)
    print("texture rgb shape", texture_rgb.shape)
    
    # Define the mesh
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    mesh = mesh.to(device)
    
    _, _, _, lights = render_cow()
    
    R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 0,
                                                     azim = np.linspace(-180, 180, num_views, endpoint=False))
    print("R shape: ", R.shape)
    print("T shape: ", T.shape)
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
    
    # Render
    rend = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    images = [image[:,:,:3] for image in rend.cpu().numpy()] 
    print("images shape: ", images[0].shape)
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    # Convert this list contatining each view into a gif
    imageio.mimsave('output/q2_1_tetrahedron.gif', images, fps=1)
    
    # Submission comment: Number of vertices in the tetrahedron mesh is 4 and number of faces is 4.
    
def q2_2(num_views):
    '''
    Construct a cube mesh and then render it from multiple viewpoints.
    Remember that we are still working with triangle meshes,
    so you will need to use two sets of triangle faces to represent one face of the cube.
    '''
    device = get_device()
    renderer = get_mesh_renderer(image_size=256)
    
    # Define the vertices and faces
    vertices = torch.tensor([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=torch.float32, device=device)
    faces = torch.tensor([[0,1,2], [0,2,3], [0,1,4], [1,4,5], [1,2,5], [2,5,6], [2,3,6], [3,6,7], [0,3,4], [3,4,7], [4,5,6], [4,6,7]], dtype=torch.int64, device=device)
    faces = faces.unsqueeze(0)
    vertices = vertices.unsqueeze(0)
    print("vertices shape: ", vertices.shape)
    print("faces shape: ", faces.shape)
    
    # Adding a single-color texture
    texture_rgb = torch.ones_like(vertices) # N X 3
    texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1.0], device=device)
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb)
    
    # Define the mesh
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    mesh = mesh.to(device)
    
    _, _, _, lights = render_cow()
    
    R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 0,
                                                        azim = np.linspace(-180, 180, num_views, endpoint=False))
    
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
    
    # Render
    rend = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights)
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Convert this list contatining each view into a gif
    imageio.mimsave('output/q2_2_cube.gif', images, fps=10)
    
    # Submission comment: Number of vertices in the cube mesh is 8 and number of faces is 12.
    
    
if __name__ == "__main__":
    q2_1(40)
    q2_2(40)