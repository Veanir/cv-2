"""
Moduł do interpretabilności modeli (Grad-CAM, Attention Maps)
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import timm
from torchvision import models
from typing import Optional, Any, List

def get_cnn_target_layer(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """
    Heurystyka do znajdowania ostatniej warstwy konwolucyjnej w popularnych architekturach CNN z `timm`.
    """
    if not hasattr(model, 'backbone'):
        print("⚠️ Ostrzeżenie: Model nie ma atrybutu 'backbone'. Nie można znaleźć warstwy dla Grad-CAM.")
        return None
        
    backbone = model.backbone
    
    # Próba 1: Standardowe nazwy bloków końcowych w `timm`
    if hasattr(backbone, 'layer4'):
        print("✅ Znaleziono warstwę docelową: 'layer4'")
        return backbone.layer4
    if hasattr(backbone, 'blocks') and isinstance(backbone.blocks, torch.nn.Sequential):
        print("✅ Znaleziono warstwę docelową: 'blocks[-1]'")
        return backbone.blocks[-1]
    if hasattr(backbone, 'features') and isinstance(backbone.features, torch.nn.Sequential):
        # Dla architektur typu DenseNet, VGG
        print("✅ Znaleziono warstwę docelową: 'features[-1]'")
        return backbone.features[-1]
    if hasattr(backbone, 'conv_head'):
         # Dla niektórych EfficientNet
         print("✅ Znaleziono warstwę docelową: 'conv_head'")
         return backbone.conv_head

    print(f"⚠️ Ostrzeżenie: Nie można automatycznie znaleźć warstwy docelowej dla {type(backbone)}. Grad-CAM nie będzie dostępny.")
    return None

class GradCAM:
    """
    Implementacja Grad-CAM do wizualizacji, na co "patrzy" model CNN.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_feature_maps))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _save_feature_maps(self, module: torch.nn.Module, input: Any, output: Any):
        self.feature_maps = output.detach()

    def _save_gradients(self, module: torch.nn.Module, grad_in: Any, grad_out: Any):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Generuje heatmapę Grad-CAM.
        """
        try:
            self.model.eval()
            
            output = self.model(x.unsqueeze(0))
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
                
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0][class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            if self.gradients is None or self.feature_maps is None:
                raise RuntimeError("Nie udało się pobrać gradientów lub map cech.")

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            
            for i in range(self.feature_maps.shape[1]):
                self.feature_maps[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
            
            return heatmap.cpu().numpy()
        finally:
            self.remove_hooks()
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

def generate_heatmap(heatmap: np.ndarray, img_size: tuple = (224, 224), colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Przetwarza surową heatmapę na obraz kolorowy."""
    heatmap = cv2.resize(heatmap, img_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    return heatmap

def overlay_heatmap(heatmap: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Nakłada heatmapę na oryginalny obraz."""
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.uint8(255 * np.clip(image, 0, 1))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(heatmap.shape) == 2:
        heatmap = generate_heatmap(heatmap, (image.shape[0], image.shape[1]))

    superimposed_img = cv2.addWeighted(heatmap, alpha, image_bgr, 1 - alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img_rgb

def get_vit_attention_map(model: torch.nn.Module, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
    """Pobiera i przetwarza mapę atencji z modelu ViT."""
    if not hasattr(model, 'get_attention_maps'):
        print("⚠️ Model nie posiada metody `get_attention_maps`. Zwracam None.")
        return None
        
    attention_map = model.get_attention_maps(image_tensor.unsqueeze(0))
    if attention_map is None:
        return None

    attention_map = attention_map.squeeze(0).cpu().numpy()
    return attention_map 