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

def q1_1(mesh, lights, num_views=12):
    '''
    create a 360-degree gif video that shows many 
    continuous views of the provided cow mesh.
    '''
    device = get_device()
    renderer = get_mesh_renderer(image_size=256)
    R, T = pytorch3d.renderer.look_at_view_transform(dist = 3, elev = 0,
                                                     azim = np.linspace(-180, 180, num_views, endpoint=False),)
    print("R shape: ", R.shape)
    print("T shape: ", T.shape)
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights) # images containing all the views
    # Convert to numpy and transpose the image dimensions
    images = [image[:,:,:3] for image in images.cpu().numpy()] 
    print("images shape: ", images[0].shape)
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    # Convert this list contatining each view into a gif
    imageio.mimsave('output/cow_views.gif', images, fps=10)
    
def q1_2():
    
    

if __name__ == "__main__":
    _, mesh, _, lights = render_cow()
    q1_1(mesh, lights)