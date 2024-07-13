import torch
import torch.nn as nn
import timm
from models.gem_pooling import GeM
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import models
import torch.nn.functional as F
import torchvision
import numpy as np


class ISICModel(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel, self).__init__()
        self.model_name = model_name
        self.model = getattr(models, model_name)(pretrained=pretrained)
        self.output_dim = self._get_output_dim()
        
        self.model.classifier = nn.Identity()
        self.model.avgpool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.output_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

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
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output