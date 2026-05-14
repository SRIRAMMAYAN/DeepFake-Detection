# CLIP wrapper module - provides basic CLIP functionality without needing the full installation
import torch
import torch.nn as nn
from torchvision import transforms
import timm

class CLIP_Wrapper:
    """Wrapper to use ViT models as CLIP alternative"""
    
    @staticmethod
    def load(model_name, device="cpu"):
        """Load a CLIP-like model using timm"""
        # Extract model name from CLIP format (e.g., "ViT-L/14" -> "vit_large")
        model_name_lower = model_name.lower()
        
        if "vit" in model_name_lower:
            if "l" in model_name_lower:
                timm_name = "vit_large_patch16_224"
            elif "b" in model_name_lower:
                timm_name = "vit_base_patch16_224"
            else:
                timm_name = "vit_base_patch16_224"
        elif "rn" in model_name_lower:
            if "50" in model_name_lower:
                timm_name = "resnet50"
            else:
                timm_name = "resnet50"
        else:
            timm_name = "vit_base_patch16_224"
        
        model = timm.create_model(timm_name, pretrained=True)
        model = model.to(device)
        model.eval()
        
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        encoder = CLIPLikeEncoder(model, device)
        return encoder, preprocess


class CLIPLikeEncoder(nn.Module):
    """Encoder that mimics CLIP's interface"""
    
    def __init__(self, model, device="cpu"):
        super().__init__()
        self.model = model
        self.device = device
        
    def encode_image(self, image):
        """Encode image to feature vector"""
        with torch.no_grad():
            features = self.model.forward_features(image)
            if hasattr(self.model, 'forward_head'):
                # For models with separate head
                features = features.mean(dim=1) if features.dim() > 2 else features
            return features
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

# Make this module act like the clip module
load = CLIP_Wrapper.load
