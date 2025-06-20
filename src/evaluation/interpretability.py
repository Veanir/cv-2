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
    print(f"🔍 Analizuję model typu: {type(model).__name__}")
    
    # Sprawdź czy to AttentionCNN
    if hasattr(model, 'backbone') and hasattr(model, 'spatial_attention'):
        print("✅ Wykryto AttentionCNN - używam backbone.forward_features")
        # Dla AttentionCNN używamy bezpośrednio backbone
        backbone = model.backbone
    elif hasattr(model, 'backbone'):
        print("✅ Wykryto standardowy model z backbone")
        backbone = model.backbone
    else:
        print("⚠️ Ostrzeżenie: Model nie ma atrybutu 'backbone'. Próbuję użyć całego modelu.")
        backbone = model
    
    print(f"Struktura backbone: {type(backbone).__name__}")
    
    # Wypisz dostępne atrybuty dla debug
    attrs = [attr for attr in dir(backbone) if not attr.startswith('_')]
    print(f"Dostępne atrybuty: {attrs[:10]}...")  # Pierwsze 10 dla przejrzystości
    
    # Próba 1: Standardowe nazwy bloków końcowych w `timm`
    if hasattr(backbone, 'layer4'):
        print("✅ Znaleziono warstwę docelową: 'layer4' (ResNet)")
        return backbone.layer4
    elif hasattr(backbone, 'blocks') and isinstance(backbone.blocks, torch.nn.Sequential):
        print("✅ Znaleziono warstwę docelową: 'blocks[-1]' (EfficientNet)")
        return backbone.blocks[-1]
    elif hasattr(backbone, 'features') and isinstance(backbone.features, torch.nn.Sequential):
        # Dla architektur typu DenseNet, VGG
        print("✅ Znaleziono warstwę docelową: 'features[-1]' (DenseNet/VGG)")
        return backbone.features[-1]
    elif hasattr(backbone, 'conv_head'):
         # Dla niektórych EfficientNet
         print("✅ Znaleziono warstwę docelową: 'conv_head' (EfficientNet)")
         return backbone.conv_head
    elif hasattr(backbone, 'stages') and isinstance(backbone.stages, torch.nn.Sequential):
         # Dla niektórych nowoczesnych architektur
         print("✅ Znaleziono warstwę docelową: 'stages[-1]'")
         return backbone.stages[-1]
    elif hasattr(backbone, 'layers') and isinstance(backbone.layers, torch.nn.Sequential):
         # Dla innych architektur
         print("✅ Znaleziono warstwę docelową: 'layers[-1]'")
         return backbone.layers[-1]
    
    # Próba uniwersalna - znajdź ostatnią warstwę Conv2d
    last_conv = None
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
            last_conv_name = name
    
    if last_conv is not None:
        print(f"✅ Znaleziono ostatnią warstwę Conv2d: '{last_conv_name}'")
        return last_conv

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
            
            # Zresetuj poprzednie wartości
            self.feature_maps = None
            self.gradients = None
            
            output = self.model(x)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
                
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0][class_idx] = 1
            output.backward(gradient=one_hot, retain_graph=True)
            
            if self.gradients is None or self.feature_maps is None:
                print(f"⚠️ Ostrzeżenie: Nie udało się pobrać gradientów lub map cech dla przykładu.")
                print(f"   Gradients: {self.gradients is not None}, Feature maps: {self.feature_maps is not None}")
                return None

            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            
            # Sprawdź czy gradients mają sens
            if torch.sum(torch.abs(pooled_gradients)) < 1e-8:
                print(f"⚠️ Ostrzeżenie: Gradients są praktycznie zero - może problem z hook'ami")
                return None
            
            # Skopiuj feature_maps aby uniknąć modyfikacji in-place
            feature_maps_copy = self.feature_maps.clone()
            for i in range(feature_maps_copy.shape[1]):
                feature_maps_copy[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(feature_maps_copy, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
            else:
                print(f"⚠️ Ostrzeżenie: Heatmap jest pusta po normalizacji")
                return None
            
            return heatmap.cpu().numpy()
            
        except Exception as e:
            print(f"❌ Błąd podczas generowania Grad-CAM: {e}")
            return None
        finally:
            # Zawsze wyczyść gradienty
            if hasattr(self.model, 'zero_grad'):
                self.model.zero_grad()
    
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
    try:
        print(f"🔍 Próbuję uzyskać attention map z modelu: {type(model).__name__}")
        
        # Sprawdź różne sposoby dostępu do attention maps
        if hasattr(model, 'get_attention_maps'):
            print("✅ Model ma metodę get_attention_maps")
            attention_map = model.get_attention_maps(image_tensor.unsqueeze(0))
            if attention_map is not None:
                attention_map = attention_map.squeeze(0).cpu().numpy()
                print(f"✅ Attention map shape: {attention_map.shape}")
                return attention_map
        
        # Alternatywny sposób dla ensemble modeli ViT
        elif hasattr(model, 'models') and len(model.models) > 0:
            print("✅ Wykryto ensemble model - próbuję pierwszy model")
            first_model = model.models[0]
            if hasattr(first_model, 'get_attention_maps'):
                attention_map = first_model.get_attention_maps(image_tensor.unsqueeze(0))
                if attention_map is not None:
                    attention_map = attention_map.squeeze(0).cpu().numpy()
                    print(f"✅ Attention map z ensemble shape: {attention_map.shape}")
                    return attention_map
        
        # Bezpośredni dostęp do ViT transformer
        elif hasattr(model, 'vit'):
            print("✅ Model ma atrybut vit - próbuję bezpośredni dostęp")
            with torch.no_grad():
                outputs = model.vit.vit(pixel_values=image_tensor.unsqueeze(0), output_attentions=True)
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    attention = outputs.attentions[-1].mean(dim=1)  # średnia z głów
                    B, S, _ = attention.shape
                    patch_size = int((S - 1) ** 0.5)  # -1 dla [CLS] token
                    attention_map = attention[:, 0, 1:].reshape(B, patch_size, patch_size)
                    attention_map = attention_map.squeeze(0).cpu().numpy()
                    print(f"✅ Direct ViT attention map shape: {attention_map.shape}")
                    return attention_map
        
        print("⚠️ Nie można uzyskać attention map - model nie obsługuje tej funkcji")
        return None
        
    except Exception as e:
        print(f"❌ Błąd podczas pobierania attention map: {e}")
        return None 