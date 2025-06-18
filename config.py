"""
Konfiguracja dla projektu Vision Transformers vs CNN w klasyfikacji medycznej
"""
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    """Konfiguracja danych"""
    data_dir: str = "data"
    dataset_name: str = "ham10000"  # HAM10000 dataset dla klasyfikacji dermatologicznej
    img_size: int = 224
    batch_size: int = 32
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 4
    
@dataclass 
class ModelConfig:
    """Konfiguracja modeli"""
    # Vision Transformer
    vit_model_name: str = "google/vit-base-patch16-224"
    vit_pretrained: bool = True
    
    # CNN Models
    cnn_model_name: str = "resnet50"  # resnet50, efficientnet_b0, densenet121
    cnn_pretrained: bool = True
    
    # Wspólne ustawienia
    num_classes: int = 7  # HAM10000 ma 7 klas
    dropout_rate: float = 0.1
    freeze_backbone: bool = False
    
@dataclass
class TrainingConfig:
    """Konfiguracja treningu"""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Augmentacja
    use_augmentation: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
@dataclass
class ExperimentConfig:
    """Konfiguracja eksperymentów"""
    experiment_name: str = "vit_vs_cnn_medical"
    output_dir: str = "results"
    save_best_model: bool = True
    log_wandb: bool = True
    
    # Testy różnych rozmiarów datasetu
    dataset_fractions: List[float] = None
    
    def __post_init__(self):
        if self.dataset_fractions is None:
            self.dataset_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
            
# Globalne instancje konfiguracji
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
experiment_config = ExperimentConfig()

# Utwórz niezbędne foldery
os.makedirs(data_config.data_dir, exist_ok=True)
os.makedirs(experiment_config.output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def check_data_ready() -> bool:
    """Sprawdza czy dane są gotowe do użycia"""
    try:
        # Import tutaj aby uniknąć circular import
        from src.data.dataset_downloader import check_ham10000_available
        return check_ham10000_available(data_config.data_dir)
    except ImportError:
        return False


def ensure_data_ready():
    """Upewnia się, że dane są gotowe lub pokazuje instrukcje"""
    if check_data_ready():
        print(f"✅ Dataset HAM10000 gotowy w {data_config.data_dir}")
        return True
    else:
        print(f"⚠️ Dataset HAM10000 nie znaleziony w {data_config.data_dir}")
        print("💡 Uruchom jeden z:")
        print("   python scripts/setup_kaggle.py  # Konfiguracja Kaggle API")
        print("   python test_kaggle_download.py  # Test pobierania")
        print("   python main.py                  # Automatyczne pobieranie")
        return False 