from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision
from torch import nn
import torch
import timm
from torchvision import models
import torch.nn.functional as F
import numpy as np


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
        # Convert 4D tensor to a list of 3D tensors
        if isinstance(images, torch.Tensor):
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            transform = torchvision.transforms.ToTensor()
            image_list = [transform(img) for img in images]
        
        with torch.no_grad():
            self.mask_rnn.eval()  # Ensure Mask R-CNN is in eval mode
            predictions = self.mask_rnn(image_list)

        segmented_images = []
        for i in range(len(predictions)):
            if predictions[i]['masks'].shape[0] > 0:
                masks = (predictions[i]['masks'] > 0.5).squeeze().cpu().numpy()
                segmented_image = np.multiply(image_list[i].cpu().numpy(), masks[0, :, :, np.newaxis])
            else:
                segmented_image = image_list[i].cpu().numpy()
            segmented_images.append(segmented_image)

        segmented_images = np.stack(segmented_images)
        return torch.from_numpy(segmented_images)


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
        
        if segmented_images.ndim == 4:
            segmented_images = segmented_images.to(images.device)  # Ensure the tensor is on the same device
        
        # Extract feature vectors from all models
        feature_vectors = [self.extract_features(segmented_images, model) for model in self.feature_extractors]
        combined_features = torch.cat(feature_vectors, dim=1).unsqueeze(0)  # Add batch dimension
        
        # Classify using GRU model
        output = self.gru_model(combined_features)
        return output