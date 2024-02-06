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
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def mesh2pcd(mesh, num_samples = 1000):
    '''
    write a function that takes a triangle mesh and the number of samples and outputs a 
    point cloud.
    1. Sample a face of a triangle in the mesh with probability proportional to the area 
    of the face
    2. Sample a random barycentric coordinate uniformly
    3. Compute the corresponding point using baricentric coordinates on the selected face.
    '''
    
    # Get the vertices and faces
    vertices, faces = mesh
    # vertices = vertices.unsqueeze(0)
    # faces = faces.unsqueeze(0)
    vertices = vertices.to(get_device())
    faces = faces.to(get_device())
    print("vertices shape: ", vertices.shape)
    print("faces shape: ", faces.shape)
    
    # Get the areas of the faces
    areas = pytorch3d.ops.mesh_face_areas_normals(vertices,faces)
    print("areas shape: ", len(areas))
    print("areas: ", areas)
    print("areas[0] shape: ", areas[0].shape)
    print("areas[1] shape: ", areas[1].shape)
    
    # Sample a face of a triangle in the mesh with probability proportional to the area of the face
    sampled_faces = torch.multinomial(areas[0], num_samples, replacement=True)
    print("sampled_faces shape: ", sampled_faces.shape)
    
    # Sample a random barycentric coordinate uniformly
    barycentric_coords = torch.rand(num_samples, 1, 3, device=get_device())
    print("barycentric_coords shape: ", barycentric_coords.shape)
    
    # Compute the corresponding point using baricentric coordinates on the selected face
    # sampled_points = pytorch3d.ops.interpolate_face_attributes(
    #     barycentric_coords, sampled_faces, vertices, faces
    # )
    sampled_points = pytorch3d.ops.interpolate_face_attributes(pix_to_face = sampled_faces.unsqueeze(0).expand(1, 1, 1, num_samples),
                                                                barycentric_coords  = barycentric_coords.unsqueeze(0).unsqueeze(0),
                                                                face_attributes = faces.unsqueeze(0).permute(1,2,0))
    print("sampled_points shape: ", sampled_points.shape)
    
    return sampled_points

if __name__ == "__main__":
    mesh = load_cow_mesh(path="data/Tree/Tree.obj")
    mesh2pcd(mesh, num_samples = 100)