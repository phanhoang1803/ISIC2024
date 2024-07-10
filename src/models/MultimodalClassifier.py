import torch
import torchvision
from torch import nn
import timm


class SingleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SingleMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu6 = nn.ReLU6()
        
    def forward(self, x):
        x = self.fc(x)
        x = self.relu6(x)
        return x

class SubsequentMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super(SubsequentMLP, self).__init__()
        
        self.mlps = nn.ModuleList()
        self.mlps.append(SingleMLP(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.mlps.append(SingleMLP(hidden_dims[i], hidden_dims[i + 1]))
        self.mlps.append(SingleMLP(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        for mlp in self.mlps:
            x = mlp(x)
        return x
        
class ImageEncoder(nn.Module):
    def __init__(self, hidden_dims: list, embed_dim: int, model_name: str="vit_l", pretrained: bool=True):
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.pretrained_image_encoder = self._create_image_encoder()
        self.pretrained_output_dim = self._get_output_dim()
        
        self.image_mlp = SubsequentMLP(self.pretrained_output_dim, hidden_dims, embed_dim)
        
    def _create_image_encoder(self):
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
        elif self.model_name.startswith("vit"):
            model = networks[self.model_name](pretrained=self.pretrained)
            model.heads = nn.Identity()
        elif self.model_name.startswith("efficientnet"):
            model = networks[self.model_name](pretrained=self.pretrained)
            model.classifier = nn.Identity()
        else:
            model = networks[self.model_name](pretrained=self.pretrained)
                    
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
            
    def forward(self, x):
        x = self.pretrained_image_encoder(x)
        x = self.image_mlp(x)
        
        return x
            
class MetadataEncoder(nn.Module):
    def __init__(self, meta_dim: int, hidden_dims: list, meta_embed_dim: int):
        super(MetadataEncoder, self).__init__()
        self.meta_mlp = SubsequentMLP(meta_dim, hidden_dims, meta_embed_dim)

    def forward(self, x):
        x = self.meta_mlp(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder, self).__init__()
        self.multihead_attn_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.multihead_attn_meta = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8)
    
    def forward(self, img_features, meta_features):
        # Multihead Attention
        img_attn_output, _ = self.multihead_attn_img(meta_features, img_features, img_features) # Q, K, V
        metadata_attn_output, _ = self.multihead_attn_meta(img_features, meta_features, meta_features) # Q, K, V
        
        # Residual connection
        img_attn_fused = img_attn_output + img_features
        meta_attn_fused = metadata_attn_output + meta_features
        
        # Fusion
        fused_features = torch.cat((img_attn_fused, meta_attn_fused), dim=-1)
        
        return fused_features
        
class MultimodalClassifier(nn.Module):
    def __init__(self, meta_dim: int, img_hidden_dims: list=[512, 256], meta_hidden_dims: list=[256, 128], embed_dim: int=64, num_classes: int=1, model_name: str="vit_l", pretrained: bool=True):
        super(MultimodalClassifier, self).__init__()
        self.image_encoder = ImageEncoder(img_hidden_dims, embed_dim, model_name, pretrained)
        self.meta_encoder = MetadataEncoder(meta_dim, meta_hidden_dims, embed_dim)
        self.decoder = Decoder(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU6(),
            nn.Linear(64, 32),
            nn.ReLU6(),
            nn.Linear(32, 1),    
        )
        
        
    def forward(self, img, meta):
        images_features = self.image_encoder(img)
        meta_features = self.meta_encoder(meta)
        
        fused_features = self.decoder(images_features, meta_features)
        output = self.fc(fused_features)
        
        return torch.nn.Sigmoid()(output)