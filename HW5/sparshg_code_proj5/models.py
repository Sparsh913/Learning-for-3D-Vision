import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        # self.layers = nn.Sequential(
        #     # nn.Linear(3, 64),
        #     # nn.ReLU(),
        #     # nn.Linear(64, 64),
        #     # nn.ReLU(),
        #     # nn.Linear(64, 128),
        #     # nn.ReLU(),
        #     # nn.Linear(128, 1024),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 512),
        #     # nn.ReLU(),
        #     # nn.Linear(512, 1024),
        #     # nn.ReLU()
        # ).to('cuda')
        self.layers = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU()
        ).to('cuda') # (B, N, 3) -> (B, N, 1024)
        # maxpooling across batch dimension to get 1024 features
        self.pool = nn.MaxPool1d(1024).to('cuda')
        self.fc1 = nn.Linear(1024, 512).to('cuda')
        self.fc2 = nn.Linear(512, 256).to('cuda')
        self.fc3 = nn.Linear(256, num_classes).to('cuda')

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # pass
        points = points.permute(0, 2, 1)
        out = self.layers(points).to(points.device) # (B, 3, N) -> (B, 1024, N)
        print("out after layers", out.shape)
        # out = self.pool(out.permute(0, 2, 1))
        # out = out.transpose(1, 2)
        
        out = torch.nn.functional.max_pool1d(out, out.shape[2]).to(points.device)
        print("out after pool", out.shape)
        out = out.permute(0, 2, 1) # (B, 1024, N) -> (B, N, 1024)
        # print("out after pool", out.shape)
        # maxpooling across batch dimension
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out).permute(0, 2, 1)
        # extract (B, num_classes) from (B, num_classes, 9)
        out = torch.max(out, dim=2)[0] # (B, num_classes, 9) -> (B, num_classes)
        # print("out after fc3", out.shape)
        return out
        

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # pass
        # self.layers1 = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        # ).to('cuda')
        self.layers1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU()
        ).to('cuda')
        
        # self.layers2 = nn.Sequential(
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1024),
        #     nn.ReLU()
        # ).to('cuda')
        self.layers2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.ReLU()
        ).to('cuda')
        # self.layers3 = nn.Sequential(
        #     nn.Linear(1088, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_seg_classes)
        # ).to('cuda')
        self.layers3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, num_seg_classes, 1)
        ).to('cuda')
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # pass
        points = points.permute(0, 2, 1)
        res1 = self.layers1(points).to(points.device) # (B, 3, N) -> (B, 64, N)
        # print("after layers1", res1.shape)
        res2 = self.layers2(res1).to(points.device) # (B, 64, N) -> (B, 1024, N)
        # max pooling across points
        res2 = nn.MaxPool1d(points.shape[2])(res2).to(points.device) # (B, 1024, N) -> (B, 1024, 1)
        # print("after maxpool", res2.shape)
        # Append res2 to res1
        res2 = res2.repeat(1, 1, points.shape[2]).to(points.device) # (B, 1024, 1) -> (B, 1024, N)
        # print("after repeat", res2.shape)
        res = torch.cat([res1, res2], dim=1).to(points.device) # (B, N, 64) + (B, 1024, N) -> (B, N, 1088)
        
        out = self.layers3(res).to(points.device) # (B, N, 1088) -> (B, N, num_seg_classes)
        # print("out", out.shape)
        return out # (B, N, num_seg_classes) -> (B, num_seg_classes, N)    
