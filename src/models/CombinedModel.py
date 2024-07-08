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


class AttentionBlock(nn.Module):
    def __init__(self, in_dim):
        """
        Initialize AttentionBlock module.

        Args:
            in_dim (int): Input dimension.
        """
        super(AttentionBlock, self).__init__()

        # Convolutional layers for query, key, and value.
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)  # (batch_size, in_dim // 8, width, height)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)  # (batch_size, in_dim // 8, width, height)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)  # (batch_size, in_dim, width, height)

        # Learnable scalar parameter gamma.
        self.gamma = nn.Parameter(torch.zeros(1))  # (1,)

    def forward(self, x):
        """
        Forward pass of the attention block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Reshape input tensor to (batch_size, channels, width * height)
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        # Compute attention weights
        attention = torch.softmax(energy, dim=-1)
        # Reshape value tensor to (batch_size, channels * width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        # Compute output tensor
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        # Apply attention weights to input tensor
        out = self.gamma * out + x
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        self.query_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        k = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        v = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        
        energy = torch.matmul(q, k)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, v).permute(0, 1, 3, 2).contiguous().view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_dim // reduction, in_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, C, width, height = x.size()
        y = self.global_avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ImageBranch(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, use_attention=False, attention_type='self-attention', num_heads=8):
        super(ImageBranch, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.cnn = self._create_cnn_model()
        self.output_dim = self._get_output_dim()
        self.use_attention = use_attention
        
        if attention_type == 'self-attention':
            self.attention = AttentionBlock(self.output_dim)
        elif attention_type == 'se':
            self.attention = SEBlock(self.output_dim)
        elif attention_type == 'multi-head':
            self.attention = MultiHeadSelfAttention(self.output_dim, num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

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
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = self.cnn.features(x)
        if self.use_attention:
            x = self.attention(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = torch.nn.Dropout(p=0.6)(x)
        return x


class MetadataAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(MetadataAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.fc_out = nn.Linear(dim, dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        energy = torch.einsum("bqhd,bkhd->bhqk", [q, k]) / self.scale
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.einsum("bhqk,bvhd->bqhd", [attention, v]).reshape(batch_size, seq_length, dim)
        out = self.fc_out(out)
        
        return out

class MetadataBranch(nn.Module):
    def __init__(self, metadata_dim, hidden_dims=[128], output_dim=64, use_attention=False, num_heads=8):
        super(MetadataBranch, self).__init__()
        self.use_attention = use_attention
        self.meta = nn.Sequential(
            # nn.Linear(metadata_dim, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            # # Swish_Module(),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.6),
            
            # nn.Linear(hidden_dims[0], output_dim),
            # nn.BatchNorm1d(output_dim),
            # # Swish_Module(),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.6),
            
            nn.Linear(metadata_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        
        if use_attention:
            self.attention = MetadataAttention(output_dim, num_heads)
    
    def forward(self, x):
        x = self.meta(x)
        
        if self.use_attention:
            # Add a sequence dimension for attention
            x = x.unsqueeze(1)
            x = self.attention(x)
            x = x.squeeze(1)
        
        return x

class CombinedModel(nn.Module):
    def __init__(self, image_model_name, metadata_dim=0, hidden_dims=[128], metadata_output_dim=32, use_attention=False, attention_type='self-attention', num_heads=8):
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
        
        self.image_branch = ImageBranch(model_name=image_model_name, 
                                        attention_type=attention_type, 
                                        use_attention=use_attention,
                                        num_heads=num_heads)
        
        # Calculate combined dimension
        combined_dim = self.image_branch.output_dim 
        
        # Initialize metadata branch if metadata_dim > 0
        if metadata_dim > 0:
            self.metadata_branch = MetadataBranch(metadata_dim=metadata_dim, 
                                                  hidden_dims=hidden_dims, 
                                                  output_dim=metadata_output_dim, 
                                                  use_attention=use_attention, 
                                                  num_heads=num_heads)
            combined_dim += metadata_output_dim
        
        # Initialize final layer
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(combined_dim, 512),  # Hidden layer
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout layer
            
            nn.Linear(256, 1),  # Hidden layer
        )
        
        # self.sigmoid = nn.Sigmoid()
    
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
        x = self.fc(x)
        
        # Because we are using BCEWithLogitsLoss,  we don't need sigmoid here
        return x
