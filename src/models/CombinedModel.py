import torch
from torch import nn
from torch import optim
from torchvision import models
import torchvision
from models.gem_pooling import GeM

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * nn.Sigmoid()(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = nn.Sigmoid()(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class ImageBranch(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(ImageBranch, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.cnn = self._create_cnn_model()
        self.output_dim = self._get_output_dim()

    def _create_cnn_model(self):
        model_architectures = {
            'resnet18': torchvision.models.resnet18,
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

        if self.model_name in ['resnet18', 'vgg16']:
            model = model_architectures[self.model_name](pretrained=self.pretrained)
            model.fc = nn.Identity()
            model.avgpool = nn.Identity()
        elif self.model_name.startswith('efficientnet_b'):
            model = model_architectures[self.model_name](pretrained=self.pretrained)
            model.classifier = nn.Identity()
            model.avgpool = nn.Identity()
        elif self.model_name == 'simple_cnn':
            model = self.__create_simple_cnn_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: resnet18, vgg16, efficientnet_b 0 to 7")

        return model

    def __create_simple_cnn_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        
        return model
    
    def _get_output_dim(self):
        lookup = {
            'resnet18': 512,
            'vgg16': 4096,
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'efficientnet_b4': 1792,
            'efficientnet_b5': 2048,
            'efficientnet_b6': 2304,
            'efficientnet_b7': 2560,
            'simple_cnn': 512
        }
        dim = lookup.get(self.model_name, None)
        if dim is None:
            raise ValueError(f"Unsupported model: {self.model_name} \n Supported models: resnet18, vgg16, efficientnet_b 0 to 7")

        return dim

    def forward(self, x):
        x = self.cnn(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        # x = torch.nn.Dropout(p=0.5)(x)
        return x

class MetadataBranch(nn.Module):
    def __init__(self, metadata_dim, hidden_dims=[128], output_dim=64):
        super(MetadataBranch, self).__init__()
        
        self.meta = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            # Swish_Module(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(hidden_dims[0], output_dim),
            nn.BatchNorm1d(output_dim),
            # Swish_Module(),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
    
    def forward(self, x):
        x = self.meta(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, image_model_name, metadata_dim=0, hidden_dims=[128], metadata_output_dim=32, num_heads=8):
        """
        Initializes the CombinedAttentionModel with the given hyperparameters.

        Args:
            image_model_name (str): The name of the image model.
            metadata_dim (int, optional): The dimension of the metadata. Defaults to 0.
            hidden_dims (list, optional): The hidden dimensions for the metadata branch. Defaults to [512, 128].
            metadata_output_dim (int, optional): The output dimension for the metadata branch. Defaults to 128.
        """
        super(CombinedModel, self).__init__()
        
        # Initialize hyperparameters
        self.image_branch = ImageBranch(model_name=image_model_name)
        combined_dim = self.image_branch.output_dim
        
        # Initialize metadata branch if metadata_dim > 0
        if metadata_dim > 0:
            self.metadata_branch = MetadataBranch(metadata_dim=metadata_dim, 
                                                  hidden_dims=hidden_dims, 
                                                  output_dim=metadata_output_dim)
            
            # self.attention_fusion = AttentionalFeatureFusion(self.image_branch.output_dim, metadata_output_dim)
            combined_dim += metadata_output_dim
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.image_branch.output_dim + metadata_output_dim,
                                                        num_heads=num_heads)
        
        # Initialize final layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(256, 1),
        )
    def forward(self, image, metadata):
        """
        Forward pass of the combined attention model.

        Args:
            image (torch.Tensor): The input image tensor.
            metadata (torch.Tensor): The input metadata tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        # Pass image through image branch and attention
        image_features = self.image_branch(image)
        
        # If metadata dimension is greater than zero, pass metadata through metadata branch and attention
        fused_features = image_features
        if self.metadata_dim > 0:
            metadata_features  = self.metadata_branch(metadata)
            
            fused_features = torch.cat([image_features, metadata_features], dim=1)
    
        
        # Reshape for MultiheadAttention input (seq_len, batch_size, embed_dim)
        fused_features = fused_features.permute(1, 0, 2)
        
        # Apply MultiheadAttention
        attn_output, _ = self.multihead_attention(fused_features, fused_features, fused_features)
        
        # Reshape output back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.permute(1, 0, 2)
        
        # Pass feature maps through final layer
        output = self.fc(attn_output)
        
        # Because we are using BCEWithLogitsLoss,  we don't need sigmoid here
        return output
