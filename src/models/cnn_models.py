"""
CNN Models dla klasyfikacji dermatologicznej
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Optional

class MedicalCNN(nn.Module):
    """
    Uniwersalna klasa CNN dla klasyfikacji medycznej, oparta na bibliotece `timm`.
    """
    
    def __init__(self,
                 model_name: str = "resnet50",
                 num_classes: int = 7,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5): # Zwiększony domyślny dropout
        """
        Args:
            model_name: Nazwa dowolnej architektury z biblioteki `timm`
            num_classes: Liczba klas
            pretrained: Czy użyć pretrenowanych wag
            freeze_backbone: Czy zamrozić backbone
            dropout_rate: Współczynnik dropout w głowicy klasyfikacyjnej
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Utwórz model bazowy za pomocą timm, bez głowicy klasyfikacyjnej
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # num_classes=0 usuwa oryginalną głowicę
        )
        
        # Pobierz wymiar cech z backbone'u
        feature_dim = self.backbone.num_features
        
        # Stwórz nową, konfigurowalną głowicę klasyfikacyjną
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Opcjonalnie zamroź backbone
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Zamraża parametry backbone'u."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Odmraża parametry backbone'u."""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        # 'features' jest już spłaszczonym tensorem cech [B, num_features]
        output = self.classifier(features)
        return output
        
    def get_feature_maps(self, x: torch.Tensor):
        """
        Zwraca mapy cech z ostatniej warstwy konwolucyjnej.
        Używa wbudowanej funkcji `timm` do ekstrakcji cech.
        """
        return self.backbone.forward_features(x)


class CNNEnsemble(nn.Module):
    """Ensemble z różnych architektur CNN"""
    
    def __init__(self,
                 model_names: list = ["resnet50", "efficientnet_b0", "densenet121"],
                 num_classes: int = 7,
                 pretrained: bool = True):
        super().__init__()
        
        self.models = nn.ModuleList([
            MedicalCNN(name, num_classes, pretrained)
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


class AttentionCNN(nn.Module):
    """CNN z mechanizmem attention"""
    
    def __init__(self,
                 base_model: str = "resnet50",
                 num_classes: int = 7,
                 pretrained: bool = True):
        super().__init__()
        
        # Base CNN
        self.backbone = MedicalCNN(base_model, num_classes, pretrained, freeze_backbone=False)
        
        # Usuń klasyfikator - zastąpimy go attention
        if base_model.startswith('resnet'):
            feature_dim = self.backbone.backbone.fc.in_features
        else:
            feature_dim = 2048  # Domyślnie
            
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 16, feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Uzyskaj cechy z backbone (bez klasyfikacji)
        features = self.backbone.backbone(x)
        
        # Zastosuj attention
        spatial_att = self.spatial_attention(features)
        channel_att = self.channel_attention(features)
        
        # Kombinuj attention
        attended_features = features * spatial_att * channel_att
        
        # Klasyfikacja
        output = self.classifier(attended_features)
        return output


def create_cnn_model(model_config: Dict[str, Any]) -> nn.Module:
    """Factory function do tworzenia modeli CNN"""
    
    model_type = model_config.get('type', 'single')
    
    if model_type == 'single':
        return MedicalCNN(
            model_name=model_config['cnn_model_name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['cnn_pretrained'],
            freeze_backbone=model_config['freeze_backbone'],
            dropout_rate=model_config['dropout_rate']
        )
    elif model_type == 'ensemble':
        return CNNEnsemble(
            model_names=model_config['model_names'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['cnn_pretrained']
        )
    elif model_type == 'attention':
        return AttentionCNN(
            base_model=model_config['cnn_model_name'],
            num_classes=model_config['num_classes'],
            pretrained=model_config['cnn_pretrained']
        )
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")


if __name__ == "__main__":
    # Test modeli
    print("Testowanie MedicalCNN...")
    
    # Test podstawowego modelu
    model = MedicalCNN(model_name="resnet50", num_classes=7)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        
    # Policz parametry
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test attention model
    print("\nTestowanie AttentionCNN...")
    att_model = AttentionCNN(num_classes=7)
    
    with torch.no_grad():
        att_output = att_model(x)
        print(f"Attention output shape: {att_output.shape}") 