"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer
from starter.utils import unproject_depth_image
import imageio


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
    cameras=None
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::1]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::1]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    # R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    # R, T = pytorch3d.renderer.look_at_view_transform(
    #     dist = 6, elev = 0, azim = np.linspace(-180, 180, 12, endpoint=False))
    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = device)
    rend = renderer(point_cloud.extend(12), cameras=cameras)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    return images


def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    print("points: ", points.shape)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

def construct_pcd(img, mask, depth, camera, path_pcd, num_views=12, path_out=None):
    '''
    use the unproject_depth_image function in utils.py to convert a depth image 
    into a point cloud (parameterized as a set of 3D coordinates and corresponding color values).
    '''
    points, rgba = unproject_depth_image(img, mask, depth, camera) # points: (N, 3), rgba: (N, 4)
    
    # make a .npz file with keys verts and rgb and save the point cloud
    np.savez(path_pcd, verts=points, rgb=rgba)
    
    # Define cameras to view the point cloud from various directions
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 6, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=-R, T=T, device = get_device())
    # Now render the point cloud using render_bridge() function
    images = render_bridge(point_cloud_path=path_pcd, image_size=256, cameras=cameras)
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)
    
    return points, rgba
    
def union_pcd(points, rgb, path_pcd, num_views=12, path_out=None):
    
    np.savez(path_pcd, verts=points, rgb=rgba)
    
    # Define cameras to view the point cloud from various directions
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 6, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=-R, T=T, device = get_device())
    # Now render the point cloud using render_bridge() function
    images = render_bridge(point_cloud_path=path_pcd, image_size=256, cameras=cameras)
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)
    
def render_torus(image_size=256, num_samples=200, device=None, num_views=12, path_out=None):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (2 + torch.cos(Theta)) * torch.cos(Phi)
    y = (2 + torch.cos(Theta)) * torch.sin(Phi)
    z = torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 6, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=-R, T=T, device = get_device())
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(torus_point_cloud.extend(num_views), cameras=cameras)
    
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)
    
def render_random(image_size=256, num_samples=200, device=None, num_views=12, path_out=None):
    """
    Renders a cube using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sqrt(torch.sin(Theta) * torch.cos(Phi) * torch.tan(Theta) * torch.sin(Phi))
    y = torch.cos(Theta)
    z = torch.sqrt(torch.sin(Theta) * torch.sin(Phi) * torch.cos(Theta) * torch.cos(Phi))

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    cube_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 6, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = get_device())
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(cube_point_cloud.extend(num_views), cameras=cameras)
    
    images = [image[:,:,:3] for image in rend.cpu().numpy()]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)
    
def render_torus_mesh(image_size=256, voxel_size=64, device=None, num_views=12, path_out=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    # Define the implicit function for a torus -> (x ** 2 + y ** 2 + z ** 2 + R ** 2 - r ** 2) ** 2 - 4 * (R ** 2) * (x ** 2 + y ** 2)
    R, r = 0.8, 0.2
    # voxels = (-torch.sqrt(X**2 + Y**2) + R)**2 + Z**2 - r**2
    voxels = (X ** 2 + Y ** 2 + Z ** 2 + R ** 2 - r ** 2) ** 2 - 4 * (R ** 2) * (X ** 2 + Y ** 2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # Define cameras to view the point cloud from various directions
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 3, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=-R, T=T, device = get_device())
    rend = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)
    # return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    images = [image[:,:,:3] for image in rend.cpu().numpy().clip(0, 1)]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)
    
def render_random_mesh(image_size=256, voxel_size=64, device=None, num_views=12, path_out=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    # voxels = torch.exp(X ** 2 - Y ** 2 - Z ** 2)
    voxels = torch.sqrt(X ** 2 + Y ** 2 - Z ** 2 - 1)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    textures = torch.randn(vertices.shape)
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    # Define cameras to view the point cloud from various directions
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist = 3, elev = 0, azim = np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device = get_device())
    rend = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)
    # return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    images = [image[:,:,:3] for image in rend.cpu().numpy().clip(0, 1)]
    
    # each image should be uint8
    images = [np.uint8(image*255) for image in images]
    
    # Save gif from the images
    imageio.mimsave(path_out, images, fps=10, loop=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    # if args.render == "point_cloud":
    #     image = render_bridge(image_size=args.image_size)
    # elif args.render == "parametric":
    #     image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
    # elif args.render == "implicit":
    #     image = render_sphere_mesh(image_size=args.image_size)
    # else:
    #     raise Exception("Did not understand {}".format(args.render))
    # plt.imsave(args.output_path, image)
    
    # Q5.1 PCD
    data = load_rgbd_data()
    # print("data: ", data)
    # Extracting rgb images, masks, depth images, and camera poses from the data dictionary
    img1, img2 = data["rgb1"], data["rgb2"]
    mask1, mask2 = data["mask1"], data["mask2"]
    depth1, depth2 = data["depth1"], data["depth2"]
    camera1, camera2 = data["cameras1"], data["cameras2"]

    # Convert img, mask, depth and camera to torch tensors
    img1, img2 = torch.tensor(img1).float(), torch.tensor(img2).float()
    mask1, mask2 = torch.tensor(mask1).float(), torch.tensor(mask2).float()
    depth1, depth2 = torch.tensor(depth1).float(), torch.tensor(depth2).float()
    # camera1 = pytorch3d.renderer.FoVPerspectiveCameras(R=camera1["R"], T=camera1["T"], device=get_device())
    
    points1, rgba1 = construct_pcd(img1, mask1, depth1, camera1, "output/q5/rgb1_pointcloud.npz", num_views=12, path_out="output/q5/pcd1.gif")
    points2, rgba2 = construct_pcd(img2, mask2, depth2, camera2, "output/q5/rgb2_pointcloud.npz", num_views=12, path_out="output/q5/pcd2.gif")
  
    # Construct a point cloud from the union of the two point clouds
    points = torch.cat([points1, points2], dim=0)
    rgba = torch.cat([rgba1, rgba2], dim=0)
    union_pcd(points, rgba, "output/q5/rgb_union_pointcloud.npz", num_views=12, path_out="output/q5/union_pcd.gif")
    
    ## Submission Note: In your submission, include a gif of each of these point clouds mentioned above side-by-side.
    
    # Q5.2 Parametric Functions
    render_torus(image_size=256, num_samples=1000, device=None, num_views=12, path_out="output/q5/torus.gif")
    render_random(image_size=256, num_samples=1000, device=None, num_views=12, path_out="output/q5/rand.gif")
    
    # Q5.3 Implicit Functions -> render a torus mesh
    render_torus_mesh(image_size=256, voxel_size=64, device=None, num_views=12, path_out="output/q5/torus_mesh.gif")
    render_random_mesh(image_size=256, voxel_size=64, device=None, num_views=12, path_out="output/q5/rand_mesh.gif")