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

def q3():
    '''
    pick 2 RGB colors, color1 and color2. We will assign the front of the cow a color 
    of color1, and the back of the cow a color of color2. The front of the cow 
    corresponds to the vertex with the smallest z-coordinate z_min, and the back of the cow 
    corresponds to the vertex with the largest z-coordinate z_max. Then, we will assign the 
    color of each vertex using linear interpolation based on the z-value of the vertex.
    '''
    device = get_device()
    renderer = get_mesh_renderer(image_size=256)
    
    vertices, faces = load_cow_mesh(path="data/cow.obj")
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    _, _, _, lights = render_cow()
    # view the cow from side
    R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 0, azim = -90)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)
    
    texture_rgb = vertices.clone()
    texture_rgb = texture_rgb.squeeze(0)
    print("texture rgb shape: ", texture_rgb.shape)
    print("vertices shape: ", vertices.shape)
    
    # Get the z-coordinates of the vertices
    z_coords = vertices[0,:,2]
    print("z_coords shape: ", z_coords.shape)
    z_min = z_coords.min()
    z_max = z_coords.max()
    print("z_min: ", z_min)
    print("z_max: ", z_max)
    
    # Define the colors
    color1 = torch.tensor([1.0, 0.0, 0.0], device=device)
    color2 = torch.tensor([0.0, 0.0, 1.0], device=device)
    
    # Linear interpolation
    t = (z_coords - z_min) / (z_max - z_min)
    t = t.unsqueeze(1)
    t = t.to(device)
    print("t shape: ", t.shape)
    print("color1 shape: ", color1.shape)
    print("color2 shape: ", color2.shape)
    
    texture_rgb = (t)*color1 + (1-t)*color2
    print("texture rgb shape: ", texture_rgb.shape)
    
    # Define the textures
    textures = pytorch3d.renderer.TexturesVertex(texture_rgb.unsqueeze(0))
    
    # Define the mesh
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
    mesh = mesh.to(device)
    
    # Render
    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    plt.imshow(rend)
    plt.show()
    # convert to uint8 and save the image
    rend = np.uint8(rend*255)
    imageio.imwrite('output/q3_cow_render.jpg', rend)
    
    # Submission comment: The front of the cow is colored red and the back of the cow is colored blue.
    
if __name__ == "__main__":
    q3()