import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
    
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        squeeze = F.adaptive_avg_pool2d(x, 1).view(batch_size, num_channels)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch_size, num_channels, 1, 1)
        return x * excitation.expand_as(x)

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, inputs):
        features = [lateral_conv(input) for input, lateral_conv in zip(inputs, self.lateral_convs)]
        for i in range(len(features) - 2, -1, -1):
            features[i] += F.interpolate(features[i + 1], scale_factor=2, mode='nearest')
        outputs = [output_conv(feature) for feature, output_conv in zip(features, self.output_convs)]
        return outputs

class EfficientNet_FPN_SE(nn.Module):
    def __init__(self):
        super(EfficientNet_FPN_SE, self).__init__()
        self.base_model = models.efficientnet_b3(pretrained=True)
        self.base_model_features = nn.Sequential(*list(self.base_model.children())[:-2])  # Remove the classifier and avgpool

        self.fpn = FPN(in_channels_list=[40, 112, 320], out_channels=256)

        self.se_blocks = nn.ModuleList([
            SEBlock(in_channels=256)
            for _ in range(3)
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        features = []
        for name, layer in self.base_model.named_children():
            x = layer(x)
            if name in ['features.3', 'features.4', 'features.6']:  # Check the correct layers
                features.append(x)

        if len(features) != 3:
            raise ValueError("Expected 3 feature maps from EfficientNet backbone, got {len(features)}")

        fpn_outputs = self.fpn(features)
        se_outputs = [se_block(output) for se_block, output in zip(self.se_blocks, fpn_outputs)]
        
        if not se_outputs:
            raise ValueError("SE outputs are empty")

        x = torch.cat(se_outputs, dim=1)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

# Example of how to instantiate and use the model
# model = ISICModel()
# print(model)

