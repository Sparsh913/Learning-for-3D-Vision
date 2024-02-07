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

def render_tree(
    tree_path="data/Tree/Tree.obj", image_size=256, color=[0.1, 0.9, 0], device=None, num_views=12, path_out="output/tree_render.gif"
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(tree_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 5, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False), at=((0, 1.5, 0),), device = get_device())
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = get_device())

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop = 50)
    
if __name__ == "__main__":
    render_tree()