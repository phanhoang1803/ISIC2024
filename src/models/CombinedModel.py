import torch
from torch import nn
from torch import optim
from torchvision import models
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
            'resnet18': models.resnet18,
            'vgg16': models.vgg16,
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7
        }
        
        model = model_architectures[self.model_name](pretrained=self.pretrained)
        
        if self.model_name in ['resnet18', 'vgg16']:
            model.fc = nn.Identity()
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.model_name.startswith('efficientnet_b'):
            model.classifier = nn.Identity()
            model.avgpool = GeM()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: resnet18, vgg16, efficientnet_b 0 to 7")
        
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
            'efficientnet_b7': 2560
        }
        dim = lookup.get(self.model_name, None) 
        if dim is None:
            raise ValueError(f"Unsupported model: {self.model_name} \n Supported models: resnet18, vgg16, efficientnet_b 0 to 7")
        
        return dim

    def forward(self, x):
        x = self.cnn(x)
        return x

class MetadataBranch(nn.Module):
    def __init__(self, metadata_dim, hidden_dims=[512], output_dim=128):
        super(MetadataBranch, self).__init__()
        self.meta = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            Swish_Module(),
            nn.Dropout(p=0.5),
            
            nn.Linear(hidden_dims[0], output_dim),
            nn.BatchNorm1d(output_dim),
            Swish_Module(),
        )
    
    def forward(self, x):
        x = self.meta(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self, image_model_name, metadata_dim=0, hidden_dims=[512, 128], metadata_output_dim=128):
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
        self.metadata_dim = metadata_dim
        
        self.image_branch = ImageBranch(model_name=image_model_name)
        
        # Calculate combined dimension
        combined_dim = self.image_branch.output_dim 
        
        # Initialize metadata branch if metadata_dim > 0
        if metadata_dim > 0:
            self.metadata_branch = MetadataBranch(metadata_dim=metadata_dim, hidden_dims=hidden_dims, output_dim=metadata_output_dim)
            combined_dim += metadata_output_dim
        
        # Initialize final layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(combined_dim, 1),  # Hidden layer
        )
        
        self.sigmoid = nn.Sigmoid()
    
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
        x = self.image_branch(image)
        
        # If metadata dimension is greater than zero, pass metadata through metadata branch and attention
        if self.metadata_dim > 0:
            x_meta = self.metadata_branch(metadata)
            x = torch.cat([x, x_meta], dim=1)
        
        # Pass feature maps through final layer
        output = self.sigmoid(self.fc(x))
        
        return output
