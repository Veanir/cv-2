"""
Moduł do pobierania i przygotowywania datasetu HAM10000
"""
import os
import requests
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import hashlib

class HAM10000Downloader:
    """Klasa do pobierania datasetu HAM10000"""
    
    # URLs dla danych (jeśli dostępne publicznie)
    METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3181890"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_metadata(self) -> bool:
        """Próbuje pobrać metadane HAM10000"""
        metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        
        # Sprawdź czy już istnieją
        if os.path.exists(metadata_path):
            print("✅ Metadane już istnieją")
            return True
            
        print("📥 Próbuję pobrać metadane HAM10000...")
        
        try:
            # Spróbuj pobrać z Harvard Dataverse
            response = requests.get(self.METADATA_URL, timeout=30)
            if response.status_code == 200:
                with open(metadata_path, 'wb') as f:
                    f.write(response.content)
                print("✅ Metadane pobrane pomyślnie!")
                return True
            else:
                print(f"⚠️ Nie udało się pobrać metadanych (status: {response.status_code})")
                
        except Exception as e:
            print(f"⚠️ Błąd podczas pobierania metadanych: {e}")
            
        return False
        
    def create_sample_data(self, num_samples: int = 1000) -> bool:
        """Tworzy przykładowe dane do testowania"""
        print(f"🔧 Tworzę {num_samples} przykładowych próbek...")
        
        # Metadane
        metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        
        if not os.path.exists(metadata_path):
            # Klasy HAM10000
            class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
            
            # Generuj przykładowe metadane
            np.random.seed(42)  # Dla reprodukowalności
            sample_data = []
            
            for i in range(num_samples):
                sample_data.append({
                    'image_id': f'ISIC_{i:07d}',
                    'dx': np.random.choice(class_names),
                    'dx_type': np.random.choice(['histo', 'follow_up', 'consensus']),
                    'age': np.random.randint(18, 85),
                    'sex': np.random.choice(['male', 'female']),
                    'localization': np.random.choice([
                        'back', 'lower extremity', 'trunk', 'upper extremity',
                        'abdomen', 'face', 'chest', 'neck', 'scalp', 'ear'
                    ])
                })
                
            df = pd.DataFrame(sample_data)
            df.to_csv(metadata_path, index=False)
            print(f"✅ Utworzono metadane: {metadata_path}")
        
        # Obrazy
        images_dir = os.path.join(self.data_dir, "HAM10000_images_part_1")
        os.makedirs(images_dir, exist_ok=True)
        
        # Sprawdź ile obrazów już mamy
        existing_images = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        if existing_images >= num_samples:
            print(f"✅ Obrazy już istnieją ({existing_images} plików)")
            return True
            
        print(f"🎨 Generuję {num_samples - existing_images} przykładowych obrazów...")
        
        # Generuj kolorowe obrazy przypominające zmiany skórne
        np.random.seed(42)
        
        for i in tqdm(range(existing_images, num_samples), desc="Tworzenie obrazów"):
            # Różne kolory dla różnych typów zmian
            base_colors = {
                'akiec': (210, 180, 140),  # Tan
                'bcc': (255, 192, 203),    # Pink
                'bkl': (222, 184, 135),    # Burlywood
                'df': (205, 133, 63),      # Peru
                'mel': (139, 69, 19),      # Saddle brown
                'nv': (160, 82, 45),       # Sienna
                'vasc': (220, 20, 60)      # Crimson
            }
            
            # Wybierz kolor bazując na klasie
            metadata = pd.read_csv(metadata_path)
            dx = metadata.iloc[i]['dx']
            base_color = base_colors.get(dx, (180, 140, 100))
            
            # Dodaj szum i wariacje
            img = Image.new('RGB', (224, 224), base_color)
            pixels = np.array(img)
            
            # Dodaj teksturę
            noise = np.random.randint(-30, 30, pixels.shape)
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            
            # Dodaj okrągłe plamy (imitujące zmiany skórne)
            center_x, center_y = np.random.randint(50, 174, 2)
            radius = np.random.randint(20, 60)
            
            y, x = np.ogrid[:224, :224]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Zmień kolor w okręgu
            spot_color = np.array(base_color) + np.random.randint(-50, 50, 3)
            spot_color = np.clip(spot_color, 0, 255)
            
            pixels[mask] = spot_color
            
            # Zapisz obraz
            img = Image.fromarray(pixels)
            img_path = os.path.join(images_dir, f"ISIC_{i:07d}.jpg")
            img.save(img_path, quality=85)
            
        print(f"✅ Utworzono {num_samples} przykładowych obrazów")
        return True
        
    def verify_data(self) -> bool:
        """Weryfikuje integralność danych"""
        print("🔍 Weryfikuję dane...")
        
        metadata_path = os.path.join(self.data_dir, "HAM10000_metadata.csv")
        
        if not os.path.exists(metadata_path):
            print("❌ Brak pliku metadanych")
            return False
            
        # Załaduj metadane
        try:
            df = pd.read_csv(metadata_path)
            print(f"✅ Metadane: {len(df)} próbek")
        except Exception as e:
            print(f"❌ Błąd w metadanych: {e}")
            return False
            
        # Sprawdź obrazy
        images_found = 0
        images_missing = 0
        
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            part_dir = os.path.join(self.data_dir, part)
            if os.path.exists(part_dir):
                images_in_part = len([f for f in os.listdir(part_dir) if f.endswith('.jpg')])
                images_found += images_in_part
                print(f"✅ {part}: {images_in_part} obrazów")
            else:
                print(f"⚠️ Brak folderu: {part}")
                
        # Sprawdź czy wszystkie obrazy z metadanych istnieją
        missing_images = []
        for _, row in df.iterrows():
            image_id = row['image_id']
            found = False
            
            for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                img_path = os.path.join(self.data_dir, part, f"{image_id}.jpg")
                if os.path.exists(img_path):
                    found = True
                    break
                    
            if not found:
                missing_images.append(image_id)
                images_missing += 1
                
        if missing_images:
            print(f"⚠️ Brakuje {len(missing_images)} obrazów")
            if len(missing_images) <= 10:
                print(f"   Przykłady: {missing_images[:5]}")
        else:
            print("✅ Wszystkie obrazy dostępne")
            
        # Podsumowanie
        coverage = (images_found - images_missing) / len(df) * 100 if len(df) > 0 else 0
        print(f"📊 Pokrycie danych: {coverage:.1f}% ({images_found - images_missing}/{len(df)})")
        
        return coverage > 50  # OK jeśli mamy >50% danych
        
    def setup_data(self, force_sample: bool = False, num_samples: int = 1000) -> bool:
        """Główna funkcja do konfiguracji danych"""
        print("🚀 Konfiguracja datasetu HAM10000...")
        
        # Sprawdź czy dane już istnieją
        if self.verify_data() and not force_sample:
            print("✅ Dane już skonfigurowane!")
            return True
            
        success = False
        
        if not force_sample:
            # Spróbuj pobrać prawdziwe dane
            print("📥 Próbuję pobrać prawdziwe dane HAM10000...")
            success = self.download_metadata()
            
            if success:
                print("ℹ️ Metadane pobrane, ale obrazy muszą być pobrane ręcznie z:")
                print("   - https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
                print("   - https://challenge.isic-archive.com/data/")
                print("   Umieść pliki w folderze:", self.data_dir)
                
        if not success or force_sample:
            # Utwórz przykładowe dane
            print("🔧 Tworzę przykładowe dane do testowania...")
            success = self.create_sample_data(num_samples)
            
        if success:
            self.verify_data()
            
        return success


def download_ham10000(data_dir: str = "data", 
                     force_sample: bool = False,
                     num_samples: int = 1000) -> bool:
    """Funkcja pomocnicza do pobierania HAM10000"""
    downloader = HAM10000Downloader(data_dir)
    return downloader.setup_data(force_sample, num_samples)


if __name__ == "__main__":
    # Test downloadera
    print("🧪 Test HAM10000 Downloader...")
    
    success = download_ham10000(
        data_dir="data_test",
        force_sample=True,
        num_samples=50
    )
    
    if success:
        print("✅ Test zakończony pomyślnie!")
    else:
        print("❌ Test nieudany!") 