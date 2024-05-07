import torch
import torch.nn as nn
import torch.nn.functional as F

# Point Transofrmer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# First define a function that outputs K nearest neighbors of the input point
def knn_points(x, k):
    # x: input point cloud data, (B, N, C)
    # k: the number of neighbors
    # compute the pairwise distance matrix (B, N, N)
    x1 = x.unsqueeze(-2) # (B, N, 1, C)
    x2 = x.unsqueeze(-3) # (B, 1, N, C)
    dist = torch.norm(x1 - x2, dim=-1) # (B, N, N)
    # get the indices of the k nearest neighbors
    _, indices = torch.topk(dist, k=k+1, largest=False, sorted=True) # (B, N, k+1)
    return indices[:, :, 1:] # exclude the query point itself

class cls_model(nn.Module):
    def __init__(self, embed_dim=64, num_classes=3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        ).to(device)
        
        self.softmax = nn.Softmax(dim=-1).to(device)
        
        self.query_proj = nn.Linear(64, embed_dim).to(device)
        self.key_proj = nn.Linear(64, embed_dim).to(device)
        self.value_proj = nn.Linear(64, embed_dim).to(device)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 2).to(device)
        
        self.linear = nn.Linear(embed_dim, num_classes).to(device)
        self.pos = PositionEncoding().to(device)
        
    def forward(self, x):
        # x: input point cloud data, (B, N, C)
        # compute the query, key, and value
        x = x.to(device)
        x = self.conv_layers(x.permute(0, 2, 1)).permute(0, 2, 1) # (B, N, 3) -> (B, N, 64)
        # print("after conv", x.shape)
        # downsample the point cloud data by a factor of 8
        x = F.max_pool1d(x.permute(0, 2, 1), 8).permute(0, 2, 1) # (B, N, 64) -> (B, N/8, 64)
        # print("after maxpool", x.shape)
        B, N, d_model = x.shape
        k = 10
        query = self.query_proj(x) # (B, N, embed_dim)
        # print("query", query.shape)
        keys = self.key_proj(x) # (B, N, embed_dim)
        # print("keys", keys.shape)
        values = self.value_proj(x) # (B, N, embed_dim)
        
        attention = torch.matmul(query, keys.transpose(-1, -2)) / d_model**0.5 # (B, N, N)
        # pos_enc = self.pos(x, k)
        attention = self.softmax(attention) # (B, N, N)
 
        idx = knn_points(x, k)
        attention = torch.gather(attention, 2, idx) # (B, N, k)
        # print("attention", attention.shape)
        
        values = torch.gather(values.unsqueeze(-2).expand(-1, -1, N, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, d_model)) # (B, N, k, embed_dim)
        # print("values", values.shape)
        
        out = attention.unsqueeze(-1).transpose(-1,-2) @ values # (B, N, 1, embed_dim)
        out = out.squeeze(-2)
        # print("out", out.shape) # (B, N, embed_dim)
        
        out = F.max_pool1d(out.permute(0,2,1), out.shape[1]) # (B, N, embed_dim) -> (B, embed_dim, 1)
        out = out.squeeze(-1) # (B, embed_dim)
        
        out = self.linear(out)
        return out
    

class PositionEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).to(device)
        
    def forward(self, x, k):
        points = x # (B, N, 64)
        B, N, d_model = points.size()
        idx = knn_points(points, k) # (B, N, k)
        neighbour_points = torch.gather(points.unsqueeze(-2).expand(-1, -1, N, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, d_model)) # (B, N, k, 64)
        points = points.unsqueeze(-2)
        diff = points - neighbour_points # (B, N, k, 64)
        diff = self.mlp(diff)
        return diff