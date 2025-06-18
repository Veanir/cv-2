"""
Vision Transformer dla klasyfikacji dermatologicznej
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from typing import Optional, Dict, Any

class MedicalViT(nn.Module):
    """Vision Transformer dostosowany do klasyfikacji medycznej"""
    
    def __init__(self, 
                 model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 7,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.1):
        """
        Args:
            model_name: Nazwa modelu ViT z HuggingFace
            num_classes: Liczba klas do klasyfikacji
            pretrained: Czy użyć pretrenowanego modelu
            freeze_backbone: Czy zamrozić backbone
            dropout_rate: Współczynnik dropout
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        if pretrained:
            # Załaduj pretrenowany model
            self.vit = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            # Utwórz model od zera
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.vit = ViTForImageClassification(config)
            
        # Dodaj dodatkowe warstwy dla lepszej adaptacji do danych medycznych
        hidden_size = self.vit.config.hidden_size
        
        # Zamiast standardowego klasyfikatora, dodaj bardziej złożony
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Opcjonalnie zamroź backbone
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Zamraża parametry backbone'u"""
        for param in self.vit.vit.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Odmraża parametry backbone'u"""
        for param in self.vit.vit.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        outputs = self.vit(pixel_values=x)
        return outputs.logits
        
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Zwraca mapy attention dla interpretacji"""
        with torch.no_grad():
            outputs = self.vit.vit(pixel_values=x, output_attentions=True)
            # Średnia z wszystkich głów attention z ostatniej warstwy
            attention = outputs.attentions[-1].mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Konwertuj na mapę 2D (pomijając [CLS] token)
            B, S, _ = attention.shape
            patch_size = int((S - 1) ** 0.5)  # -1 dla [CLS] token
            
            attention_map = attention[:, 0, 1:].reshape(B, patch_size, patch_size)
            return attention_map


class ViTEnsemble(nn.Module):
    """Ensemble z różnych rozmiarów ViT"""
    
    def __init__(self, 
                 model_names: list = ["google/vit-base-patch16-224", "google/vit-large-patch16-224"],
                 num_classes: int = 7,
                 pretrained: bool = True):
        super().__init__()
        
        self.models = nn.ModuleList([
            MedicalViT(name, num_classes, pretrained) 
            for name in model_names
        ])
        
        # Warstwa łącząca predykcje
        self.fusion = nn.Linear(len(model_names) * num_classes, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass przez wszystkie modele"""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        # Konkatenuj predykcje
        combined = torch.cat(predictions, dim=1)
        
        # Końcowa predykcja
        output = self.fusion(combined)
        return output


def create_vit_model(model_config: Dict[str, Any]) -> nn.Module:
    """Factory function do tworzenia modeli ViT"""
    
    model_type = model_config.get('type', 'single')
    
    if model_type == 'single':
        return MedicalViT(
            model_name=model_config['vit_model_name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['vit_pretrained'],
            freeze_backbone=model_config['freeze_backbone'],
            dropout_rate=model_config['dropout_rate']
        )
    elif model_type == 'ensemble':
        return ViTEnsemble(
            model_names=model_config['model_names'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['vit_pretrained']
        )
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")


if __name__ == "__main__":
    # Test modelu
    print("Testowanie MedicalViT...")
    
    model = MedicalViT(num_classes=7)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Test attention maps
        attention = model.get_attention_maps(x)
        print(f"Attention shape: {attention.shape}")
        
    # Policz parametry
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}") 