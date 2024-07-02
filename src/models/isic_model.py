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
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

class ModifiedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ModifiedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.swish_relu = lambda x: x * torch.sigmoid(x)  # Swish activation

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        out = self.swish_relu(out)
        return out

class ISICModel_MaskRNN_GRU(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None):
        super(ISICModel_MaskRNN_GRU, self).__init__()
        
        # Load Mask R-CNN model for segmentation
        self.mask_rnn = maskrcnn_resnet50_fpn(pretrained=True).eval()
        
        # Load feature extractors
        self.feature_extractors = nn.ModuleList([
            self._load_feature_extractor('resnext101_32x8d'),
            self._load_feature_extractor('xception'),
            self._load_feature_extractor('inception_v3'),
            self._load_feature_extractor('efficientnet_b0')
        ])
        
        # Define GRU model for classification
        self.gru_model = ModifiedGRU(input_size=4*2048, hidden_size=512, num_classes=num_classes)
        
    def _load_feature_extractor(self, model_name):
        if model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the classifier
        elif model_name == 'xception':
            model = timm.create_model('xception', pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the classifier
        elif model_name == 'inception_v3':
            model = models.inception_v3(pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the classifier
        elif model_name == 'efficientnet_b0':
            model = timm.create_model('efficientnet_b0', pretrained=True)
            model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the classifier
        return model
    
    def segment_image(self, images):
        if not isinstance(images, torch.Tensor):
            transform = torchvision.transforms.ToTensor()
            image_tensor = transform(images).unsqueeze(0)  # Add batch dimension if not already there
        else:
            image_tensor = images.unsqueeze(0)  # Add batch dimension if not already there

        with torch.no_grad():
            self.mask_rnn.eval()  # Ensure Mask R-CNN is in eval mode
            predictions = self.mask_rnn(image_tensor)

        if predictions[0]['masks'].shape[0] > 0:
            masks = (predictions[0]['masks'] > 0.5).squeeze().cpu().numpy()
            segmented_image = np.multiply(images.cpu().numpy(), masks[0, :, :, np.newaxis])
        else:
            segmented_image = image_tensor.squeeze().cpu().numpy()

        return segmented_image

    def extract_features(self, images, model):
        if not isinstance(images, torch.Tensor):
            transform = torchvision.transforms.ToTensor()
            images = torch.stack([transform(img) for img in images])

        with torch.no_grad():
            features = model(images)
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        return features

    def forward(self, images):
        # Segment the images
        segmented_images = self.segment_image(images)
        
        # Extract feature vectors from all models
        feature_vectors = [self.extract_features(segmented_images, model) for model in self.feature_extractors]
        combined_features = torch.cat(feature_vectors, dim=1).unsqueeze(0)  # Add batch dimension
        
        # Classify using GRU model
        output = self.gru_model(combined_features)
        return output