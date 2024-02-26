import pytorch3d
from utils import get_mesh_renderer, get_points_renderer, get_device
import torch
import imageio
import numpy as np

def mesh_to_image(
    mesh, color = [0.7, 0.7, 1], num_views=12, 
    out_path=None, dist = 65, elev = 0, at = ((0, 0, 0),)
    ):
    device = get_device()
    renderer = get_mesh_renderer(image_size=512, device=device)
    try:
        vertices, faces = mesh.verts_packed(), mesh.faces_packed()
    except:
        vertices, faces = mesh
    print("vertices shape: ", vertices.shape)
    # print("faces shape: ", faces.shape)
    print("type of faces: ", type(faces))
    vertices = torch.tensor(vertices, dtype=torch.float32).to(device)
    # convert faces to numpy float32
            # faces = faces.verts_idx
        # faces = faces.astype(np.float32)
    # faces = faces.astype(np.float32)
    try:
        # faces = faces.cpu().numpy().astype(np.float32)
        faces = faces.astype(np.float32)
        faces = torch.tensor(faces, dtype=torch.float32).to(device)
    except:
        pass
    faces = torch.tensor(faces, dtype=torch.float32).to(device)
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    # Mean center the vertices
    # vertices = vertices - vertices.mean(1)
    vertices -= vertices.mean(1, keepdim=True)
    # Create a textures object
    textures = torch.ones_like(vertices, device=device, dtype=torch.float32)
    textures = textures * torch.tensor(color, dtype=torch.float32).to(device)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    # Prepare the cameras
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = dist, elev = elev, azim = np.linspace(-180, 180, num_views, endpoint=False), at= at)
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    images = renderer(mesh.extend(num_views), cameras=many_cameras, lights=lights) # images containing all the views
    # Convert to numpy and transpose the image dimensions
    images = [image[:,:,:3] for image in images.cpu().numpy()] 
    print("images shape: ", images[0].shape)
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    # Convert this list contatining each view into a gif
    imageio.mimsave(out_path + '.gif', images, fps=10, loop=5000)
    print("Gif saved at: ", out_path + '.gif')
    
def points_to_image(pcd, num_views=12, out_path=None, image_size=256, background_color=(1, 1, 1),):
    
    device = get_device()
    
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts = torch.Tensor(pcd.squeeze(0)).to(device).unsqueeze(0)
    # rgb = torch.Tensor(pcd["rgb"][::1]).to(device).unsqueeze(0)
    rgb = torch.zeros_like(verts, device=device, dtype=torch.float32)
    pcd = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    
    # Setup the cameras
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 2, elev = -5,azim = np.linspace(-180, 180, num_views, endpoint=False), at=((0, 0, 0),))
    many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(device = device, R=R, T=T)
    rend = renderer(pcd.extend(12), cameras = many_cameras)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    images = [image[:,:,:3] for image in rend.cpu().detach().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    imageio.mimsave(out_path + '.gif', images, fps=10, loop=5000)
    print("Gif saved at: ", out_path + '.gif')