import torch
import torch.nn as nn
import timm
from models.gem_pooling import GeM
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import models
import torch.nn.functional as F
import torchvision
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = nn.ReLU6()
        
        self.dropout = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.fc = nn.Linear(in_features, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.model(images)
        x = self.pooling(x).flatten(1)
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
                
        out /= len(self.dropouts)
        
        return out
    
