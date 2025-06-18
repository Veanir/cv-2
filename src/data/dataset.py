"""
Moduł do obsługi datasetu HAM10000 dla klasyfikacji dermatologicznej
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import zipfile
from typing import Tuple, Optional, Dict, List

class HAM10000Dataset(Dataset):
    """Dataset dla HAM10000 - klasyfikacja zmian skórnych"""
    
    # Mapowanie klas
    CLASS_NAMES = [
        'akiec',  # Actinic keratoses and intraepithelial carcinomae
        'bcc',    # Basal cell carcinoma
        'bkl',    # Benign keratosis-like lesions
        'df',     # Dermatofibroma
        'mel',    # Melanoma
        'nv',     # Melanocytic nevi
        'vasc'    # Vascular lesions
    ]
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: Ścieżka do folderu z danymi
            split: 'train', 'val' lub 'test'
            transform: Transformacje obrazów
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Załaduj przygotowane splity
        self.metadata = self._load_prepared_splits()
        
        # Filtruj dane dla odpowiedniego split'a
        self.data = self.metadata[self.metadata['split'] == self.split].reset_index(drop=True)
        
        if len(self.data) == 0:
            raise RuntimeError(f"Nie znaleziono danych dla splitu '{self.split}'. "
                               "Uruchom `python scripts/prepare_dataset.py`.")

        # Mapuj etykiety na indeksy
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASS_NAMES)}
        
    def _load_prepared_splits(self) -> pd.DataFrame:
        """Ładuje wstępnie przygotowany plik z podziałami danych."""
        splits_path = os.path.join(self.data_dir, "ham10000_splits.csv")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(
                f"Plik podziału '{splits_path}' nie został znaleziony.\n"
                "Proszę najpierw uruchomić skrypt: python scripts/prepare_dataset.py"
            )
        return pd.read_csv(splits_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Zwraca obraz, etykietę oraz ID obrazu."""
        row = self.data.iloc[idx]
        label = row['dx']
        image_path = os.path.join(self.data_dir, row['path'])
        image_id = row['image_id']
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")
            
        # Załaduj obraz
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Konwertuj etykietę na indeks
        label_idx = self.class_to_idx[label]
        
        return image, label_idx, image_id
    

def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """Zwraca transformacje obrazów dla danego split'a"""
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(data_dir: str, 
                      batch_size: int = 32, 
                      img_size: int = 224,
                      num_workers: int = 4,
                      download_if_needed: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoader'y dla train, val i test"""
    
    # Sprawdź czy dane są gotowe (pobrane i przygotowane)
    splits_file = os.path.join(data_dir, 'ham10000_splits.csv')
    if not os.path.exists(splits_file):
        print("⚠️ Plik podziału danych nie istnieje.")
        if download_if_needed:
            print("   Próbuję automatycznie pobrać i przygotować dane...")
            from .dataset_downloader import download_ham10000
            from scripts.prepare_dataset import create_splits
            
            # Krok 1: Pobieranie
            if download_ham10000(data_dir):
                # Krok 2: Przygotowanie
                create_splits(data_dir)
            else:
                raise RuntimeError("Automatyczne pobieranie i przygotowanie danych nie powiodło się.")
        else:
            raise FileNotFoundError(f"Brak pliku {splits_file}. Uruchom `scripts/prepare_dataset.py`.")

    # Utwórz datasety
    train_dataset = HAM10000Dataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train', img_size)
    )
    
    val_dataset = HAM10000Dataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val', img_size)
    )
    
    test_dataset = HAM10000Dataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test', img_size)
    )
    
    # Utwórz DataLoader'y
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test datasetu
    print("Testowanie datasetu HAM10000...")
    
    # Uruchomienie testowe wymaga automatycznego przygotowania
    train_loader, val_loader, test_loader = create_dataloaders("data", batch_size=4, download_if_needed=True)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test jednej paczki
    for images, labels, image_ids in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Image IDs: {image_ids}")
        break 