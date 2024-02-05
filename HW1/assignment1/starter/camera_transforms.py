"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from starter.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    T_relative= ([0, 0, 0],[0,0,0], [0,0,2], [0.6, -0.5, 0], [0, 0, 0]),
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    print("Size of R_relative: ", len(R_relative))
    print("Size of T_relative: ", len(T_relative))
    
    R_rel = torch.tensor(R_relative).float()
    T_rel = torch.tensor(T_relative).float()
    
    # Find R_relative to see the object from the side, that is rotate by 90 degrees about y axis
    R_relative = torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    
    for R_relative, T_relative in zip(R_rel, T_rel):
        
        R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
        # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
        # we need to add R.t() to compensate that.
        renderer = get_mesh_renderer(image_size=image_size)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
        rend = renderer(meshes, cameras=cameras, lights=lights)
        image = rend[0, ..., :3].cpu().numpy()
        plt.imshow(image)
        plt.show()
        # plt.imsave(f"output/q4/transformation_{}.jpg", image)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="output/q4/transformation_1.jpg")
    args = parser.parse_args()

    # plt.imsave(args.output_path, render_cow(cow_path=args.cow_path, image_size=args.image_size))
    render_cow()