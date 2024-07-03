import torch
from torch import nn
from torch import optim
from torchvision import models

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
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
            model.fc = nn.Identity()  # Remove the final classification layer
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=self.pretrained)
            model.classifier[-1] = nn.Identity() 
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
            model.classifier = nn.Identity() 
        else:
            raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: resnet18, vgg16, efficientnet_b0")
        
        return model

    def _get_output_dim(self):
        if self.model_name == 'resnet18':
            return 512
        elif self.model_name == 'vgg16':
            return 4096
        elif self.model_name == 'efficientnet_b0':
            return 1280
        else:
            raise ValueError(f"Unsupported model: {self.model_name} \n Supported models: resnet18, vgg16, efficientnet_b0")

    def forward(self, x):
        x = self.cnn(x)
        return x

class MetadataBranch(nn.Module):
    def __init__(self, metadata_dim, hidden_dims=[512, 128], output_dim=128):
        super(MetadataBranch, self).__init__()
        self.meta = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            Swish_Module(),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            Swish_Module(),
            nn.Dropout(p=0.3),
            
            nn.Linear(hidden_dims[1], output_dim),
            nn.BatchNorm1d(output_dim),
            Swish_Module(),
        )
    
    def forward(self, x):
        x = self.meta(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            Swish_Module(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        attn_applied = torch.mul(x, attn_weights)
        return attn_applied, attn_weights


class CombinedAttentionModel(nn.Module):
    def __init__(self, image_model_name, metadata_dim=0, hidden_dims=[512, 128], metadata_output_dim=128):
        """
        Initializes the CombinedAttentionModel with the given hyperparameters.

        Args:
            image_model_name (str): The name of the image model.
            metadata_dim (int, optional): The dimension of the metadata. Defaults to 0.
            hidden_dims (list, optional): The hidden dimensions for the metadata branch. Defaults to [512, 128].
            metadata_output_dim (int, optional): The output dimension for the metadata branch. Defaults to 128.
        """
        super(CombinedAttentionModel, self).__init__()
        
        # Initialize hyperparameters
        self.metadata_dim = metadata_dim
        
        self.image_branch = ImageBranch(model_name=image_model_name)
        self.image_attention = Attention(input_dim=self.image_branch.output_dim)
        
        # Calculate combined dimension
        combined_dim = self.image_branch.output_dim 
        
        # Initialize metadata branch if metadata_dim > 0
        if metadata_dim > 0:
            self.metadata_branch = MetadataBranch(metadata_dim=metadata_dim, hidden_dims=hidden_dims, output_dim=metadata_output_dim)
            self.metadata_attention = Attention(input_dim=metadata_output_dim)
            combined_dim += metadata_output_dim
        
        # Initialize final layer
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),  # Hidden layer
            Swish_Module(),  # Activation function
            nn.Dropout(p=0.3),  # Dropout layer
            nn.Linear(256, 1)  # Output layer
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
        x = self.image_branch(image)
        x, image_attn_weights = self.image_attention(x)
        
        # If metadata dimension is greater than zero, pass metadata through metadata branch and attention
        if self.metadata_dim > 0:
            x_meta = self.metadata_branch(metadata)
            x_meta, metadata_attn_weights = self.metadata_attention(x_meta)
            # Concatenate image and metadata feature maps
            x = torch.cat([x, x_meta], dim=1)
        
        # Pass feature maps through final layer
        output = nn.Sigmoid(self.fc(x))
        
        return output
