import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .core import PointNetfeat


class PointGen(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGen, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


class PointGenComp(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenComp, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 500 * 3)
        self.encoder = PointNetfeat(num_points=2000)
        self.th = nn.Tanh()

    def forward(self, x, noise):
        batchsize = x.size()[0]
        x, _ = self.encoder(x)
        x = torch.cat([x, noise], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, 500)
        return x


class PointGenComp2(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenComp2, self).__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2500 * 3)
        self.encoder = PointNetfeat(num_points=2000)
        self.th = nn.Tanh()

    def forward(self, x, noise):
        batchsize = x.size()[0]
        x, _ = self.encoder(x)
        x = torch.cat([x, noise], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, 2500)
        return x


class PointGenR(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenR, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 500 * 3)
        self.lstm = nn.LSTM(input_size=20, hidden_size=100, num_layers=2)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[1]
        x, _ = self.lstm(x)
        x = x.view(-1, 100)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.th(self.fc4(x))

        x = x.view(5, batchsize, 1500)

        x = x.transpose(1, 0).contiguous()
        x = x.view(batchsize, 7500)

        x = x.view(batchsize, 3, 2500)
        return x


class PointGenR2(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenR2, self).__init__()

        self.decoder = nn.Sequential(nn.Linear(100, 256), nn.ReLU(),
                                     nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, 1024), nn.ReLU(),
                                     nn.Linear(1024, 500 * 3), nn.Tanh())

        self.lstmcell = nn.LSTMCell(input_size=100, hidden_size=100)

        self.encoder = nn.Sequential(PointNetfeat(num_points=500))
        self.encoder2 = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024,
                                                                      512),
                                      nn.BatchNorm1d(512), nn.ReLU(),
                                      nn.Linear(512, 100),)

    def forward(self, x):
        batchsize = x.size()[0]
        outs = []
        out = self.decoder(x)
        out = out.view(batchsize, 3, 500)
        outs.append(out)

        hx = Variable(torch.zeros(batchsize, 100))
        cx = Variable(torch.zeros(batchsize, 100))
        if x.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()

        for i in range(4):
            hd, _ = self.encoder(outs[-1])
            hd = self.encoder2(hd)
            hx, cx = self.lstmcell(hd, (hx, cx))

            out = self.decoder(hx)
            out = out.view(batchsize, 3, 500)
            outs.append(out)

        x = torch.cat(outs, 2)

        return x


class PointGenR3(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenR3, self).__init__()

        def get_decoder():
            return nn.Sequential(nn.Linear(200, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 500 * 3),
                                 nn.Tanh())
        self.decoder = get_decoder()

        self.lstmcell = nn.LSTMCell(input_size=100, hidden_size=100)

        self.encoder = nn.Sequential(PointNetfeat(num_points=500))
        self.encoder2 = nn.Sequential(nn.BatchNorm1d(1024), nn.Linear(1024,
                                                                      512),
                                      nn.BatchNorm1d(512), nn.ReLU(),
                                      nn.Linear(512, 100),)

    def forward(self, x):
        batchsize = x.size()[0]

        hx = Variable(torch.zeros(batchsize, 100))
        cx = Variable(torch.zeros(batchsize, 100))

        outs = []

        if x.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()

        for i in range(5):
            if i == 0:
                hd = Variable(torch.zeros(batchsize, 100))
            else:
                hd, _ = self.encoder(torch.cat(outs, 2))
                hd = self.encoder2(hd)
            if x.is_cuda:
                hd = hd.cuda()

            hx, cx = self.lstmcell(hd, (hx, cx))

            out = self.decoder(torch.cat([hx, x[:, :, i]], 1))
            out = out.view(batchsize, 3, 500)
            outs.append(out)

        x = torch.cat(outs, 2)

        return x


class PointGenC(nn.Module):
    def __init__(self, num_points=2500):
        super(PointGenC, self).__init__()
        self.conv1 = nn.ConvTranspose1d(100, 1024, 2, 2, 0)
        self.conv2 = nn.ConvTranspose1d(1024, 512, 5, 5, 0)
        self.conv3 = nn.ConvTranspose1d(512, 256, 5, 5, 0)
        self.conv4 = nn.ConvTranspose1d(256, 128, 2, 2, 0)
        self.conv5 = nn.ConvTranspose1d(128, 64, 5, 5, 0)
        self.conv6 = nn.ConvTranspose1d(64, 3, 5, 5, 0)

        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.th = nn.Tanh()

    def forward(self, x):

        x = x.view(-1, 100, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        x = self.th(x)
        return x


class PointGenPSG(nn.Module):
    def __init__(self, num_points=2048):
        super(PointGenPSG, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points / 4 * 3 * 1)
        self.th = nn.Tanh()

        self.conv1 = nn.ConvTranspose2d(100, 1024, (2, 3))
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        batchsize = x.size()[0]

        x1 = x
        x2 = x

        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, self.num_points / 4 * 1)

        x2 = x2.view(-1, 100, 1, 1)
        x2 = F.relu((self.conv1(x2)))
        x2 = F.relu((self.conv2(x2)))
        x2 = F.relu((self.conv3(x2)))
        x2 = F.relu((self.conv4(x2)))
        x2 = self.th((self.conv5(x2)))

        x2 = x2.view(-1, 3, 32 * 48)

        return torch.cat([x1, x2], 2)
