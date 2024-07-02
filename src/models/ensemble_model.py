import torch
import torch.nn as nn
import torchvision.models as models

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        # Load pre-trained models
        self.resnet = models.resnet50(pretrained=True)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)

        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(2048 + 1280 + 147456, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Define activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Extract features
        resnet_features = self.resnet(x).view(x.size(0), -1)
        efficientnet_features = self.efficientnet(x).view(x.size(0), -1)
        densenet_features = self.densenet(x).view(x.size(0), -1)
        
        print(resnet_features.shape, efficientnet_features.shape, densenet_features.shape)
        print(resnet_features.shape[1] + efficientnet_features.shape[1] + densenet_features.shape[1])
        
        # Concatenate features
        features = torch.cat((resnet_features, efficientnet_features, densenet_features), dim=1)
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(features))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply sigmoid activation for binary classification
        output = self.sigmoid(x)
        return output