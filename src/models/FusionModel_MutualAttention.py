import torch
import torchvision
from torch import nn
import timm

class ImageEncoder(nn.Module):
    def __init__(self, model_name="vit_l", pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_encoder = self._create_image_encoder(model_name)
        self.output_dim = self._get_output_dim()
        
    def _create_image_encoder(self, model_name):
        networks = {
            "vit_l": torchvision.models.vit_l_32,
            "vit_b": torchvision.models.vit_b_32,
            "densenet121": torchvision.models.densenet121,
            "swin_b": torchvision.models.swin_b,
            "nest_base": timm.create_model("nest_base", pretrained=self.pretrained),
        }
        
        model = torchvision.models.vit_l_32(pretrained=self.pretrained)
        
        if model_name == "nest_base":
            model = networks[model_name]
            model.head = nn.Identity()
            
        else:
            model = networks[model_name](pretrained=self.pretrained)
            model.head = nn.Identity()
            
        return model
        #     # model.eval()
        # else:
        #     raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: vit_l")
    
    
    def _get_output_dim(self):
        lookup = {
            "vit_l": 1024,
            "vit_b": 768,
            "densenet121": 1024,
            "swin_b": 1536,
            "nest_base": 1024
        }
        
        dim = lookup.get(self.model_name, None)
        if dim is None:
            raise ValueError(f"Unsupported model: {self.model_name}\n Supported models: vit_l, vit_b, densenet121, swin_b, nest_base")
        
        return dim
            
    def forward(self, x):
        x = self.image_encoder(x)
        
        return x
            

class FusionModel_MutualAttention(nn.Module):
    def __init__(self, model_name="vit_l", pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        
        self.image_encoder = 