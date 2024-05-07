import torch
import torch.nn as nn
import torch.nn.functional as F
from models import cls_model

# implement the pointnet++ model

def furthest_point_sample(x, n_c):
    """
    :param x: input point cloud [B, N, C]
    :param n_c: number of cent
    :return: cent [B, n_c, C]
    """
    B, N, C = x.shape
    cent = torch.zeros(B, n_c, C).to(x.device)
    dist = torch.ones(B, N).to(x.device) * 1e8
    far = torch.randint(0, N, (B,)).to(x.device)
    batch_idx = torch.arange(B).to(x.device)
    for i in range(n_c):
        cent[:, i, :] = x[batch_idx, far, :]
        centroid = cent[:, i, :].unsqueeze(1)
        dist = torch.sum((x - centroid) ** 2, -1)
        mask = dist < dist
        dist[mask] = dist[mask]
        far = torch.max(dist, -1)[1]
    return cent

def knn(x, y, k):
    dist = torch.cdist(y,x)
    _, indices = torch.topk(dist, k, largest=False)
    return indices

def grouping_layer(x, cent, k):
    """
    :param x: input point cloud [B, N, C]
    :param cent: cent [B, n_c, C]
    :param k: number of neighbors
    :return: neighbors [B, n_c, k, C]
    """
    B, N, C = x.shape
    n_c = cent.shape[1]
    indices = knn(x, cent, k)
    # print("indices shape: ", indices.shape)
    # for every centroid select k neighbors, can use gather
    neighbors = torch.gather(x.unsqueeze(1).expand(B, n_c, N, C), 2, indices.unsqueeze(-1).expand(B, n_c, k, C))
    return neighbors
    
class cls_model(nn.Module):
    def __init__(self, num_classes = 3):
        super(cls_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # print("x shape: ", x.shape)
        B, N, C = x.shape
        k = 32
        n_c = 1024
        cent = furthest_point_sample(x, n_c)
        # print("cent shape: ", cent.shape)
        neighbors = grouping_layer(x, cent, k)
        # print("neighbors shape: ", neighbors.shape) # [B, n_c, k, C]
        x = neighbors.permute(0, 3, 1, 2) # [B, C, n_c, k]
        x = x.reshape(B, C, -1) # [B, C, n_c*k]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # print("x shape after conv: ", x.shape)
        x = x.reshape(B, -1, n_c*k) # [B, 1024, n_c*k]
        # x = torch.max(x, -1)[0]
        x = F.max_pool1d(x, x.shape[2]).squeeze(-1)
        # print("x shape after pool: ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x