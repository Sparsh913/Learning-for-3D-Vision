import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import pytorch3d.io
from starter.utils import *
from pytorch3d.io import load_obj
from pytorch3d.vis.plotly_vis import plot_scene
# from starter.render_mesh import render_cow

# device = torch.device("cpu")
device = get_device()

renderer = get_mesh_renderer(image_size=512)
vertices, faces = load_cow_mesh(path="data/cow.obj")
print("vertices shape: ", vertices.shape)
print("face_props shape: ", faces.shape)
# textures
print("vertices: ", vertices[0])
print("faces: ", faces[0])

fmin, fmax = faces.min(), faces.max()
print("faces min: ", fmin)
print("faces max: ", fmax)

# Batched vertices and faces
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
print("vertices shape: ", vertices.shape)
print("faces shape: ", faces.shape)

# Adding texture -> Color RGB
texture_rgb = torch.ones_like(vertices) # N X 3
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1.0])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb) # important
print("texture rgb", texture_rgb)
print("texture rgb shape", texture_rgb.shape)

# Define the mesh
mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)
print("mesh device: ", mesh.device)

# define the camera
R = torch.eye(3).unsqueeze(0) # (1, 3, 3)
T = torch.tensor([[0,0,3.0]]) # (1, 3)
print("T shape: ", T.shape)
fov = 60
cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T, fov=fov)
print("cameras device: ", cameras.device)

# Lights
lights = pytorch3d.renderer.PointLights(device = device, location=[[0.0, 0.0, -3.0]])
print("lights device: ", lights.device)

# Transfer all the objects to cpu
mesh = mesh.to(device)
cameras = cameras.to(device)
lights = lights.to(device)

# Render
rend = renderer(device = device, meshes_world=mesh, cameras=cameras, lights=lights)
print("rend device: ", rend.device)
print("rend shape: ", rend.shape)
rend = rend.to(torch.device("cpu"))

plot_scene({
    "figure": {
        "Mesh": mesh,
        "Camera": cameras,
    }
})
img = rend[0, ..., :3].numpy()
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()