#!/usr/bin/env python3
"""
Demo script dla nowego systemu pobierania HAM10000 z Kaggle (tylko .env)
"""
import os
import sys
from pathlib import Path

# Dodaj src do path
sys.path.append(str(Path(__file__).parent / "src"))

from data.dataset_downloader import HAM10000Downloader, download_ham10000, HAM10000DatasetError
from data.dataset import create_dataloaders

def demo_kaggle_download():
    """Demonstracja pobierania z Kaggle używając .env"""
    print("🚀 Demo: Pobieranie HAM10000 z Kaggle (.env)")
    print("=" * 50)
    
    print("\n🎯 Prawdziwe dane z Kaggle API")
    print("-" * 40)
    
    try:
        downloader = HAM10000Downloader("data")
        
        print("\n📋 Instrukcje konfiguracji Kaggle (.env):")
        downloader.show_setup_instructions()
        print("\n⏸️ Jeśli nie masz skonfigurowanego Kaggle, uruchom:")
        print("   python scripts/setup_kaggle.py")
        print("\n▶️ Próbuję pobrać prawdziwe dane...")
        
        # Pobierz dane
        success = downloader.setup_data()
        
        if success:
            print("✅ Prawdziwe dane pobrane pomyślnie!")
            
            # Test loadera danych
            print("🧪 Testowanie DataLoader...")
            try:
                train_loader, val_loader, test_loader = create_dataloaders(
                    data_dir="data",
                    batch_size=4
                )
                
                print(f"   📊 Train: {len(train_loader.dataset)} próbek")
                print(f"   📊 Val: {len(val_loader.dataset)} próbek")
                print(f"   📊 Test: {len(test_loader.dataset)} próbek")
                
                # Test jednej paczki
                for images, labels in train_loader:
                    print(f"   🖼️ Batch shape: {images.shape}")
                    print(f"   🏷️ Labels shape: {labels.shape}")
                    break
                    
                print("✅ DataLoader działa poprawnie z prawdziwymi danymi!")
                
            except Exception as e:
                print(f"❌ Błąd DataLoader: {e}")
                
    except HAM10000DatasetError as e:
        print("\n" + str(e))
        print("\n⚠️ To jest oczekiwane zachowanie - potrzebujesz konfiguracji Kaggle API!")
        
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd: {e}")
    
    # Podsumowanie
    print(f"\n📋 Podsumowanie:")
    print("=" * 50)
    print("✅ System wymaga prawdziwych danych HAM10000!")
    print("✅ Używa TYLKO Kaggle API z kluczami z .env!")
    print("✅ NIE ma fallback'ów ani alternatywnych źródeł!")
    print("✅ Jasno komunikuje co trzeba zrobić!")
    print()
    print("🎯 Następne kroki:")
    print("1. Skonfiguruj Kaggle API: python scripts/setup_kaggle.py")
    print("2. Lub ręcznie utwórz plik .env z KAGGLE_USERNAME i KAGGLE_KEY")
    print("3. Uruchom eksperymenty: python main.py")
    print("4. System pobierze prawdziwe dane HAM10000!")

def analyze_dataset_structure():
    """Analizuje strukturę datasetu"""
    print("\n🔍 Analiza struktury datasetu HAM10000")
    print("=" * 50)
    
    # Sprawdź czy mamy jakiekolwiek dane
    data_dirs = ["data"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"\n📁 Folder: {data_dir}")
            
            # Metadane
            metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
            if os.path.exists(metadata_path):
                import pandas as pd
                df = pd.read_csv(metadata_path)
                print(f"   📊 Metadane: {len(df)} próbek")
                
                print("   ✅ Prawdziwe dane HAM10000")
                
                # Rozkład klas
                class_counts = df['dx'].value_counts()
                print(f"   📈 Rozkład klas:")
                for class_name, count in class_counts.items():
                    percentage = count / len(df) * 100
                    print(f"      {class_name}: {count} ({percentage:.1f}%)")
            
            # Obrazy
            for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                part_path = os.path.join(data_dir, part)
                if os.path.exists(part_path):
                    image_count = len([f for f in os.listdir(part_path) if f.endswith('.jpg')])
                    print(f"   🖼️ {part}: {image_count} obrazów")

def check_env_config():
    """Sprawdza konfigurację .env"""
    print("\n🔧 Sprawdzanie konfiguracji .env")
    print("=" * 40)
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ Brak pliku .env")
        print("💡 Utwórz plik .env z:")
        print("   KAGGLE_USERNAME=twoj_username")
        print("   KAGGLE_KEY=twój_klucz_api")
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if username and key:
            print(f"✅ Znaleziono konfigurację dla: {username}")
            print("✅ Klucz API jest ustawiony")
            return True
        else:
            print("❌ Brak KAGGLE_USERNAME lub KAGGLE_KEY w .env")
            return False
            
    except ImportError:
        print("❌ Brak pakietu python-dotenv")
        print("   Zainstaluj: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Błąd sprawdzania .env: {e}")
        return False

def main():
    """Główna funkcja demo"""
    print("🧪 Test systemu pobierania HAM10000 - TYLKO Kaggle API (.env)")
    print("=" * 60)
    
    # Sprawdź konfigurację .env
    check_env_config()
    
    # Demo pobierania
    demo_kaggle_download()
    
    # Analiza struktury
    analyze_dataset_structure()
    
    print("\n🎉 Demo zakończone!")
    print("💡 System używa TYLKO Kaggle API - to jest bezpieczne i rzetelne!")

if __name__ == "__main__":
    main() 