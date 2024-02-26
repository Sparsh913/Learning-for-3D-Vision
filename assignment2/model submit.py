from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            
            # TODO:
            # self.decoder = 
            '''
            Define a neural network that takes in a 512-dimensional feature vector 
            and outputs a 32x32x32 voxel grid.
            https://arxiv.org/pdf/1603.08637.pdf
            '''
            # b x 512 -> b x 32 x 32 x 32
            self.layer1 = nn.Linear(512, 4096)
            self.layer2 = nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU()
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU()
            )
            
            self.layer4 = nn.Sequential(
                nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU()
            )
            
            self.layer5 = nn.Sequential(
                nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU()
            )
            
            self.layer6 = nn.Sequential(
                nn.Conv3d(8, 1, kernel_size=1),
                nn.Sigmoid()
            )
            
            
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # https://arxiv.org/pdf/1612.00603.pdf
            # Let's try the vanilla predictor as the decoder
            # self.decoder = nn.Sequential(
            #     nn.Linear(512, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, self.n_point),
            #     nn.ReLU(),
            #     nn.Linear(self.n_point, self.n_point*3),
            #     # nn.Tanh()
            # )
            
            self.layer1 = nn.Sequential(
                nn.Linear(512, 512)
            )
            self.act1 = nn.LeakyReLU()
            self.layer2 = nn.Sequential(
                nn.Linear(512, 512)
            )
            self.act2 = nn.LeakyReLU()
            self.layer3 = nn.Sequential(
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.n_point*3)
            )
                        
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(2, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            self.decoder = nn.Sequential(
                nn.Linear(512, mesh_pred.verts_packed().shape[0]*3),
                nn.Tanh()
            )
                            

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.layer1(encoded_feat)
            voxels_pred = voxels_pred.view(-1, 512, 2, 2, 2) # b x 256 x 2 x 2 x 2
            voxels_pred = self.layer2(voxels_pred)
            voxels_pred = self.layer3(voxels_pred)
            voxels_pred = self.layer4(voxels_pred)
            voxels_pred = self.layer5(voxels_pred)
            voxels_pred = self.layer6(voxels_pred)
            print("voxels_pred shape", voxels_pred.shape)
            
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred = self.decoder(encoded_feat)
            # pointclouds_pred = pointclouds_pred.view(-1, self.n_point, 3)           
            # return pointclouds_pred
            
            pointclouds_pred = self.layer1(encoded_feat)
            reidue = pointclouds_pred
            pointclouds_pred = self.act1(pointclouds_pred)
            pointclouds_pred = self.layer2(pointclouds_pred)
            pointclouds_pred += reidue
            pointclouds_pred = self.act2(pointclouds_pred)
            pointclouds_pred = self.layer3(pointclouds_pred)
            pointclouds_pred = pointclouds_pred.view(-1, self.n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred

