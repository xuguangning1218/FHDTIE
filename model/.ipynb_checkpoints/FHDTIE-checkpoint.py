import torch
from torch import nn
import torchvision.models as models
from .UNet import UNet
from .GCN import GCN

class FHDTIE(nn.Module):
    def __init__(self, config):
        super(FHDTIE, self).__init__()
        self.in_channel = int(config['model']['in_channel'])
        self.r_in_channel = int(config['model']['r_in_channel'])
        self.gcn_hidden1 = int(config['model']['gcn_hidden1'])
        self.gcn_hidden2 = int(config['model']['gcn_hidden2'])
        self.gcn_fnn_hidden = int(config['model']['gcn_fnn_hidden'])
        self.num_cluster = int(config['model']['num_cluster'])
        
        # backbone
        self.resnet_cnn = models.resnet18(num_classes=1)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(self.in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )
        
        self.resnet_cnn = nn.Sequential(*( list(self.resnet_cnn.children())[4:-2] ))
        self.pooling = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*8*8, 512),
            # nn.ReLU(inplace=True),
        )

        # RSI

        # shape macther
        self.reanalysis_feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.r_in_channel, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # channel fuser
        self.fusion = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        # FCFI
        self.unet = UNet(in_channels=self.in_channel, out_channels=self.num_cluster)
        

        self.gcn1 = GCN(in_features=self.gcn_hidden1, out_features=self.gcn_hidden2)
        self.gcn2 = GCN(in_features=self.gcn_hidden2, out_features=self.gcn_fnn_hidden)
        self.glinear = nn.Linear(in_features=self.gcn_fnn_hidden*self.num_cluster, out_features=self.gcn_fnn_hidden, bias=True)
        
        self.regressor = nn.Linear(in_features=512+self.gcn_fnn_hidden, out_features=1, bias=True)


    def forward(self, x, r_data, adj):
        b, c, h, w = x.shape

        clus_label, clus_feature = self.unet(x, r_data)

        gout = self.gcn1(clus_feature, adj)
        gout = self.gcn2(gout, adj)
        gout = gout.reshape(b, -1)
        gout = self.glinear(gout)
        
        out = self.feature_extractor(x)
        r = self.reanalysis_feature_extractor(r_data)
        fusion = torch.cat([out, r], dim=1)
        out = self.fusion(fusion)
        out = self.resnet_cnn(out)
        out = self.pooling(out)
        out = torch.cat([out, gout], dim=1)
        out = self.regressor(out)
        
        return out, clus_label