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
                 transform: Optional[transforms.Compose] = None,
                 download: bool = True):
        """
        Args:
            data_dir: Ścieżka do folderu z danymi
            split: 'train', 'val' lub 'test'
            transform: Transformacje obrazów
            download: Czy pobrać dane jeśli nie istnieją
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        if download:
            self._download_data()
            
        # Załaduj metadane
        self.metadata = self._load_metadata()
        
        # Filtruj dane dla odpowiedniego split'a
        self.data = self._prepare_split()
        
        # Mapuj etykiety na indeksy
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASS_NAMES)}
        
    def _download_data(self):
        """Pobiera i przygotowuje dane HAM10000"""
        from .dataset_downloader import download_ham10000
        
        print("🚀 Sprawdzam i konfiguruję dane HAM10000...")
        
        # Użyj nowego downloadera
        success = download_ham10000(
            data_dir=self.data_dir,
            force_sample=False,  # Najpierw spróbuj prawdziwych danych
            num_samples=500      # Większy rozmiar dla lepszych podziałów
        )
        
        if not success:
            print("❌ Nie udało się skonfigurować danych!")
            raise RuntimeError("Konfiguracja danych nieudana")
        

        
    def _load_metadata(self) -> pd.DataFrame:
        """Ładuje metadane datasetu"""
        metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        return pd.read_csv(metadata_path)
        
    def _prepare_split(self) -> pd.DataFrame:
        """Przygotowuje podział danych na train/val/test"""
        # Stratified split zachowujący proporcje klas
        np.random.seed(42)
        
        grouped = self.metadata.groupby('dx')
        train_data = []
        val_data = []
        test_data = []
        
        for class_name, group in grouped:
            n_samples = len(group)
            indices = np.random.permutation(n_samples)
            
            train_end = int(0.7 * n_samples)
            val_end = int(0.85 * n_samples)
            
            if self.split == 'train':
                train_data.append(group.iloc[indices[:train_end]])
            elif self.split == 'val':
                val_data.append(group.iloc[indices[train_end:val_end]])
            elif self.split == 'test':
                test_data.append(group.iloc[indices[val_end:]])
                
        if self.split == 'train':
            return pd.concat(train_data, ignore_index=True)
        elif self.split == 'val':
            return pd.concat(val_data, ignore_index=True)
        else:
            return pd.concat(test_data, ignore_index=True)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Zwraca obraz i etykietę"""
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['dx']
        
        # Znajdź ścieżkę do obrazu
        image_path = None
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            potential_path = os.path.join(self.data_dir, part, f"{image_id}.jpg")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
                
        if image_path is None:
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_id}")
            
        # Załaduj obraz
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Konwertuj etykietę na indeks
        label_idx = self.class_to_idx[label]
        
        return image, label_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """Oblicza wagi klas dla niezbalansowanego datasetu"""
        class_counts = self.data['dx'].value_counts()
        total_samples = len(self.data)
        
        weights = []
        for class_name in self.CLASS_NAMES:
            if class_name in class_counts:
                weight = total_samples / (len(self.CLASS_NAMES) * class_counts[class_name])
            else:
                weight = 1.0
            weights.append(weight)
            
        return torch.FloatTensor(weights)


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
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tworzy DataLoader'y dla train, val i test"""
    
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
    
    train_loader, val_loader, test_loader = create_dataloaders("data", batch_size=4)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Test jednej paczki
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break 