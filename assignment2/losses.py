import torch
# import pytorch3d
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops.knn import knn_points, knn_gather

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	# loss = torch.mean(torch.abs(voxel_src-voxel_tgt))
	# assert voxel_src.shape == voxel_tgt.shape
	# Note that voxel_src contains the logits and not the probabilities
	# convert logits to probabilities
	voxel_src = torch.sigmoid(voxel_src) # shape -> b x h x w x d
	voxel_tgt = torch.sigmoid(voxel_tgt)
	# flatten the voxel grid
	# voxel_src = voxel_src.view(-1)
	# voxel_tgt = voxel_tgt.view(-1)
	# calculate the binary cross entropy loss
	# loss = torch.nn.functional.binary_cross_entropy(voxel_src,voxel_tgt) # shape -> scalar
	# maximize log-likelihood of GT label in each voxel -> (1/N)sum(pn * log(pn_hat) + (1-pn) * log(1-pn_hat))
	loss = -torch.mean(voxel_tgt * torch.log(voxel_src) + (1-voxel_tgt) * torch.log(1-voxel_src))  # shape -> scalar
	# loss = voxel_tgt * torch.log(voxel_src) + (1-voxel_tgt) * torch.log(1-voxel_src)  # shape -> b x h x w x d
	# # mean over the batch
	# loss = torch.mean(loss) 
	# for i in range(voxel_src.shape[0]):
	# 	loss += 
	# convert loss to a tensor
	# loss = torch.tensor(loss, requires_grad=True)
	# print("requires_grad loss: ", loss.requires_grad)
	# print("requires_grad voxel_src: ", voxel_src.requires_grad)
	return loss

# def chamfer_loss(point_cloud_src,point_cloud_tgt):
# 	# point_cloud_src, point_cloud_src: b x n_points x 3  
# 	# loss_chamfer = 
# 	# implement chamfer loss from scratch
# 	# loss_chamfer = sum(min_i(||x_i - y_j||^2) + min_j(||x_i - y_j||^2))
# 	# loss_chamfer = torch.mean(torch.sum(torch.min(torch.sum((point_cloud_src.unsqueeze(2) - point_cloud_tgt.unsqueeze(1))**2, dim=3), dim=1), dim=1))
# 	# flatten the point clouds
# 	# point_cloud_src = point_cloud_src.view(point_cloud_src.shape[0], -1,3)
# 	# point_cloud_tgt = point_cloud_tgt.view(point_cloud_tgt.shape[0], -1,3)
	
# 	# compute the distance matrix
# 	# dist_matrix_src_tgt = torch.sum((point_cloud_src.unsqueeze(2) - point_cloud_tgt.unsqueeze(1))**2, dim=-1) # shape -> b x n_points_src x n_points_tgt
# 	src2tgt = torch.cdist(point_cloud_src, point_cloud_tgt)
# 	tgt2src = torch.cdist(point_cloud_tgt, point_cloud_src)
# 	# compute the chamfer distance
# 	chamfer_distance_src, _ = torch.min(src2tgt, 1)
# 	chamfer_distance_tgt, _ = torch.min(tgt2src, 1)
# 	loss_chamfer = torch.mean(chamfer_distance_src) + torch.mean(chamfer_distance_tgt)
# 	# mean over the batch
# 	loss_chamfer = torch.mean(loss_chamfer) # shape -> scalar
# 	return loss_chamfer

# def chamfer_loss(point_cloud_src,point_cloud_tgt):
# 	# point_cloud_src, point_cloud_src: b x n_points x 3  
# 	# loss_chamfer = 
# 	# implement chamfer loss from scratch
# 	point_cloud_src_cpy, point_cloud_tgt_cpy = point_cloud_src, point_cloud_tgt
# 	n_src = torch.full((point_cloud_src_cpy.shape[0],), point_cloud_src_cpy.shape[1], dtype=torch.int64, device=point_cloud_src.device)
# 	n_tgt = torch.full((point_cloud_tgt_cpy.shape[0],), point_cloud_tgt_cpy.shape[1], dtype=torch.int64, device=point_cloud_tgt.device)
	
# 	src_nn = knn_points(point_cloud_src, point_cloud_tgt, lengths1=n_src, lengths2=n_tgt, norm=2, K=1)
# 	tgt_nn = knn_points(point_cloud_tgt, point_cloud_src, lengths1=n_tgt, lengths2=n_src, norm=2, K=1)

# 	cham_x = src_nn.dists[..., 0].sum(1)
# 	cham_y = tgt_nn.dists[..., 0].sum(1)

# 	loss_chamfer = torch.mean(cham_x + cham_y)
	
# 	return loss_chamfer

def chamfer_loss(point_cloud_src,point_cloud_tgt):
    # point_cloud_src, point_cloud_src: b x n_points x 3
    
    dist_src, dist_tgt = torch.cdist(point_cloud_src,point_cloud_tgt), torch.cdist(point_cloud_tgt,point_cloud_src)
    #find the minimum distance for each point in point_cloud_src
    min_distance_src, _ = torch.min(dist_src,1)
    min_distance_tgt, _ = torch.min(dist_tgt,1)

    loss_chamfer = torch.mean(min_distance_src) + torch.mean(min_distance_tgt)
    return loss_chamfer

# def chamfer_loss(point_cloud_src, point_cloud_tgt):
# 	# point_cloud_src, point_cloud_src: b x n_points x 3  
# 	p1_dists, _, _ = knn_points(point_cloud_src, point_cloud_tgt)
# 	p2_dists, _, _ = knn_points(point_cloud_tgt, point_cloud_src)
# 	# implement chamfer loss from scratch
# 	loss_chamfer = torch.sum((p1_dists + p2_dists)) / point_cloud_src.shape[0]
# 	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	# loss_laplacian = torch.mean(torch.sum((mesh_src.verts_packed() - mesh_src.verts_packed().mean(dim=0))**2, dim=1)) # shape -> scalar
	loss_laplacian = mesh_laplacian_smoothing(mesh_src) # shape -> scalar
	return loss_laplacian