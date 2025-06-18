"""
Moduł do pobierania i przygotowywania datasetu HAM10000 z Kaggle
"""
import os
import pandas as pd
from PIL import Image
from typing import Optional
from dotenv import load_dotenv

class HAM10000DatasetError(Exception):
    """Wyjątek dla problemów z pobraniem datasetu HAM10000"""
    pass

class HAM10000Downloader:
    """Klasa do pobierania datasetu HAM10000 z Kaggle"""
    
    # Kaggle dataset info
    KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Załaduj zmienne środowiskowe z .env
        load_dotenv()
        
    def _check_kaggle_credentials(self) -> bool:
        """Sprawdza czy Kaggle API jest skonfigurowane w .env"""
        try:
            kaggle_username = os.getenv('KAGGLE_USERNAME')
            kaggle_key = os.getenv('KAGGLE_KEY')
            
            if kaggle_username and kaggle_key:
                print("✅ Kaggle credentials znalezione w .env")
                return True
            else:
                print("⚠️ Brak KAGGLE_USERNAME lub KAGGLE_KEY w pliku .env")
                return False
                
        except Exception as e:
            print(f"⚠️ Błąd sprawdzania Kaggle credentials: {e}")
            return False
    
    def download_from_kaggle(self) -> bool:
        """Pobiera dataset HAM10000 z Kaggle"""
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            if not self._check_kaggle_credentials():
                return False
                
            print("📥 Pobieranie HAM10000 z Kaggle...")
            
            # Ustaw credentials z .env
            os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
            os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
            
            # Inicjalizuj API
            api = KaggleApi()
            api.authenticate()
            
            # Pobierz dataset
            api.dataset_download_files(
                self.KAGGLE_DATASET,
                path=self.data_dir,
                unzip=True,
                quiet=False
            )
            
            print("✅ Dataset pobrany z Kaggle!")
            
            # Sprawdź czy pliki istnieją
            expected_files = [
                "HAM10000_metadata.csv",
            ]
            
            missing_files = []
            for file in expected_files:
                if not os.path.exists(os.path.join(self.data_dir, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"⚠️ Brakuje plików: {missing_files}")
                print("   Sprawdź czy pobieranie z Kaggle się powiodło")
                return False
                
            # Sprawdź czy mamy foldery z obrazami
            image_folders = []
            for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
                folder_path = os.path.join(self.data_dir, folder)
                if os.path.exists(folder_path):
                    count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                    image_folders.append(f"{folder}: {count} obrazów")
                    print(f"✅ {folder}: {count} obrazów")
                    
            if not image_folders:
                print("⚠️ Nie znaleziono folderów z obrazami")
                print("   Dataset może być w innym formacie")
                return False
                
            return True
            
        except ImportError:
            print("❌ Brak pakietu 'kaggle'")
            print("   Zainstaluj: pip install kaggle")
            return False
        except Exception as e:
            print(f"❌ Błąd pobierania z Kaggle: {e}")
            return False
        
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
        
        # OK jeśli mamy >90% danych (tolerujemy kilka brakujących plików)
        return coverage > 90
        
    def setup_data(self) -> bool:
        """Główna funkcja do konfiguracji danych - TYLKO Kaggle API"""
        print("🚀 Konfiguracja datasetu HAM10000...")
        
        # Sprawdź czy dane już istnieją
        if self.verify_data():
            print("✅ Dataset HAM10000 już jest pobrany i gotowy do użycia!")
            print(f"📁 Lokalizacja: {self.data_dir}")
            return True
            
        print("📥 Dataset nie został znaleziony, pobieranie z Kaggle...")
        # Pobierz z Kaggle (jedyna opcja)
        success = self.download_from_kaggle()
        
        # Jeśli nie udało się - BŁĄD!
        if not success:
            error_msg = """
❌ BŁĄD: Nie udało się pobrać datasetu HAM10000 z Kaggle!

🔧 Rozwiązanie:

1. KAGGLE API (.env):
   - Utwórz plik .env w głównym katalogu projektu
   - Dodaj do pliku .env:
     KAGGLE_USERNAME=twoj_username
     KAGGLE_KEY=twoj_api_key
   
2. UZYSKAJ KLUCZE KAGGLE:
   - Utwórz konto na https://www.kaggle.com
   - Idź do Account → API → Create New API Token
   - Pobierz plik kaggle.json
   - Skopiuj username i key z tego pliku do .env

3. DATASET: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

⚠️  UWAGA: Ten projekt używa TYLKO Kaggle API!
    Nie ma backup'ów ani alternatywnych źródeł danych.
""".format(data_dir=self.data_dir)
            
            raise HAM10000DatasetError(error_msg)
            
        # Weryfikuj po pobraniu
        if not self.verify_data():
            raise HAM10000DatasetError("Dataset pobrany, ale weryfikacja nie powiodła się!")
            
        return True

    def show_setup_instructions(self):
        """Pokazuje instrukcje konfiguracji"""
        print("🔧 Konfiguracja Kaggle API:")
        print()
        print("1. Utwórz konto na https://www.kaggle.com")
        print("2. Idź do Account → API → Create New API Token")
        print("3. Pobierz plik kaggle.json")
        print("4. Utwórz plik .env w głównym katalogu projektu")
        print("5. Dodaj do .env:")
        print("   KAGGLE_USERNAME=twoj_username")
        print("   KAGGLE_KEY=twoj_api_key")
        print()
        print("Przykład .env:")
        print("KAGGLE_USERNAME=jankowalski")
        print("KAGGLE_KEY=abc123def456...")
        print()


def download_ham10000(data_dir: str = "data") -> bool:
    """Funkcja pomocnicza do pobierania HAM10000 - TYLKO Kaggle API"""
    downloader = HAM10000Downloader(data_dir)
    return downloader.setup_data()


def check_ham10000_available(data_dir: str = "data") -> bool:
    """Sprawdza czy dataset HAM10000 jest dostępny (bez pobierania)"""
    downloader = HAM10000Downloader(data_dir)
    return downloader.verify_data()


if __name__ == "__main__":
    # Test downloadera
    print("🧪 Test HAM10000 Downloader z Kaggle...")
    
    # Pokaż instrukcje konfiguracji
    downloader = HAM10000Downloader("data")
    downloader.show_setup_instructions()
    
    try:
        success = download_ham10000(data_dir="data")
        
        if success:
            print("✅ Test zakończony pomyślnie!")
        else:
            print("❌ Test nieudany!")
            
    except HAM10000DatasetError as e:
        print("\n" + str(e))
        print("\n💡 Skonfiguruj Kaggle API w pliku .env aby pobrać dane!") 