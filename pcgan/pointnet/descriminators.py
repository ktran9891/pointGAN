import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .core import PointNetfeat


class PointNetCls(nn.Module):
    def __init__(self, num_points=2500, k=2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x), trans


class PointDecoder(nn.Module):
    def __init__(self, num_points=2048, k=2):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


class PointNetAE(nn.Module):
    def __init__(self, num_points=2048, k=2):
        super(PointNetAE, self).__init__()
        self.num_points = num_points
        self.encoder = nn.Sequential(PointNetfeat(num_points,
                                                  global_feat=True,
                                                  trans=False),
                                     nn.Linear(1024, 256), nn.ReLU(),
                                     nn.Linear(256, 100))

        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PointNetReg(nn.Module):
    def __init__(self, num_points=2500, k=1):
        super(PointNetReg, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, trans


class PointNetReg2(nn.Module):
    def __init__(self, num_points=500, k=3):
        super(PointNetReg2, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)
        self.fc4 = nn.Linear(100, k)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x1, trans1 = self.feat(x1)
        x2, trans2 = self.feat(x2)

        x = torch.cat([x1, x2], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x, trans1, trans2


class PointNetDenseCls(nn.Module):
    def __init__(self, num_points=2500, k=2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k))
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans
