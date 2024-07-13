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
        self.model_name = model_name
        self.pretrained = pretrained
        self.output_dim = self._get_output_dim()
        
        self.cnn = self._create_cnn()
        self.pooling = nn.AdaptiveAvgPool2d(1)        
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.7) for _ in range(5)])
        
        self.fc = nn.Linear(self.output_dim, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def _create_cnn(self):
        networks = {
            "vit_l": torchvision.models.vit_l_32,
            "vit_b": torchvision.models.vit_b_32,
            "densenet121": torchvision.models.densenet121,
            "swin_b": torchvision.models.swin_b,
            'resnet18': torchvision.models.resnet18,
            'resnet50': torchvision.models.resnet50,
            'vgg16': torchvision.models.vgg16,
            'efficientnet_b0': torchvision.models.efficientnet_b0,
            'efficientnet_b1': torchvision.models.efficientnet_b1,
            'efficientnet_b2': torchvision.models.efficientnet_b2,
            'efficientnet_b3': torchvision.models.efficientnet_b3,
            'efficientnet_b4': torchvision.models.efficientnet_b4,
            'efficientnet_b5': torchvision.models.efficientnet_b5,
            'efficientnet_b6': torchvision.models.efficientnet_b6,
            'efficientnet_b7': torchvision.models.efficientnet_b7
        }
        
        # model = torchvision.models.vit_l_32(pretrained=self.pretrained)
        if self.model_name == "nest_base":
            model = timm.create_model("nest_base", pretrained=self.pretrained),
        else:
            model = networks[self.model_name](pretrained=self.pretrained)
                    
        if self.model_name in ['resnet18', 'resnet50', 'vgg16']:
            model.fc = nn.Identity()
        elif self.model_name.startswith("vit"):
            model.heads = nn.Identity()
        elif self.model_name.startswith("efficientnet"):
            model.classifier = nn.Identity()               
            model.avgpool = nn.AdaptiveAvgPool2d(1)     
                    
        return model
    
    def _get_output_dim(self):
        lookup = {
            "vit_l": 1024,
            "vit_b": 768,
            "densenet121": 1024,
            "swin_b": 1536,
            "nest_base": 1024,
            'resnet18': 512,
            'resnet50': 2048,
            'vgg16': 4096,
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560,
            
        }
        
        dim = lookup.get(self.model_name, None)
        if dim is None:
            raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: vit_l, vit_b, densenet121, swin_b, nest_base")
        
        return dim

    def forward(self, images):
        x = self.cnn(images)
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
                
        out /= len(self.dropouts)
        
        return out
    
