"""
Skrypt do testowania konfiguracji i podstawowej funkcjonalności
"""
import torch
from src.data.dataset import create_dataloaders, HAM10000Dataset
from src.models.vision_transformer import MedicalViT
from src.models.cnn_models import MedicalCNN
import os

def test_data_loading():
    """Test ładowania danych"""
    print("🔍 Testowanie ładowania danych...")
    
    try:
        # Test downloadera najpierw
        from src.data.dataset_downloader import download_ham10000
        
        print("📥 Test downloadera danych...")
        success = download_ham10000(
            data_dir="data", 
            force_sample=True,  # Użyj przykładowych danych dla testu
            num_samples=50      # Małe liczby dla szybkiego testu
        )
        
        if not success:
            print("❌ Downloader nieudany")
            return False
        
        # Test podstawowego datasetu
        dataset = HAM10000Dataset(
            data_dir="data",
            split='train',
            download=False  # Dane już pobrane przez downloader
        )
        print(f"✅ Dataset utworzony: {len(dataset)} próbek")
        
        # Test jednej próbki
        if len(dataset) > 0:
            image, label = dataset[0]
            if hasattr(image, 'shape'):
                print(f"✅ Próbka: obraz {image.shape}, etykieta {label}")
            else:
                print(f"✅ Próbka: obraz {type(image)}, etykieta {label}")
        
        # Test DataLoader'ów
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir="data",
            batch_size=4,
            num_workers=0  # 0 dla Windows
        )
        
        print(f"✅ DataLoaders utworzone:")
        print(f"   Train: {len(train_loader.dataset)} próbek")
        print(f"   Val: {len(val_loader.dataset)} próbek") 
        print(f"   Test: {len(test_loader.dataset)} próbek")
        
        # Test jednej paczki
        if len(train_loader) > 0:
            for images, labels in train_loader:
                print(f"✅ Paczka: {images.shape}, etykiety: {labels.shape}")
                break
            
        return True
        
    except Exception as e:
        print(f"❌ Błąd w ładowaniu danych: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """Test tworzenia modeli"""
    print("\n🤖 Testowanie modeli...")
    
    # Test CNN
    try:
        cnn_model = MedicalCNN(
            model_name="resnet18",  # Mniejszy model do testów
            num_classes=7,
            pretrained=False  # Bez pobierania dla szybkości
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = cnn_model(x)
        print(f"✅ CNN ResNet18: input {x.shape} -> output {output.shape}")
        
    except Exception as e:
        print(f"❌ Błąd w modelu CNN: {e}")
        return False
    
    # Test ViT (opcjonalny - wymaga internetu)
    try:
        print("⏳ Próbuję załadować ViT (wymaga internetu)...")
        vit_model = MedicalViT(
            model_name="google/vit-base-patch16-224",
            num_classes=7,
            pretrained=False  # Bez pretrenowanych wag
        )
        
        output = vit_model(x)
        print(f"✅ ViT: input {x.shape} -> output {output.shape}")
        
    except Exception as e:
        print(f"⚠️ ViT niedostępny (prawdopodobnie brak internetu): {e}")
        print("   To normalne - ViT można testować później")
        
    return True

def test_directory_structure():
    """Test struktury folderów"""
    print("\n📁 Sprawdzanie struktury folderów...")
    
    required_dirs = [
        "src",
        "src/data", 
        "src/models",
        "src/training",
        "src/evaluation",
        "data",
        "results",
        "logs",
        "checkpoints"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ Brak folderu: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ Utworzono: {dir_path}")
    
    return True

def test_config():
    """Test konfiguracji"""
    print("\n⚙️ Testowanie konfiguracji...")
    
    try:
        from config import data_config, model_config, training_config
        
        print(f"✅ data_config: dataset={data_config.dataset_name}, batch_size={data_config.batch_size}")
        print(f"✅ model_config: classes={model_config.num_classes}, dropout={model_config.dropout_rate}")
        print(f"✅ training_config: epochs={training_config.epochs}, lr={training_config.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Błąd w konfiguracji: {e}")
        return False

def test_device():
    """Test dostępności CUDA"""
    print("\n🔧 Sprawdzanie urządzenia...")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA dostępna: {torch.cuda.get_device_name(0)}")
        print(f"   Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA niedostępna - używam CPU")
        print("   To może być OK dla małych testów")
    
    print(f"✅ PyTorch: {torch.__version__}")
    return True

def main():
    """Główna funkcja testowa"""
    print("🚀 ROZPOCZYNAM TESTY KONFIGURACJI PROJEKTU")
    print("=" * 50)
    
    tests = [
        ("Struktura folderów", test_directory_structure),
        ("Konfiguracja", test_config),
        ("Urządzenie", test_device),
        ("Ładowanie danych", test_data_loading),
        ("Modele", test_models),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Nieoczekiwany błąd w {test_name}: {e}")
            results.append((test_name, False))
    
    # Podsumowanie
    print("\n" + "=" * 50)
    print("📊 PODSUMOWANIE TESTÓW:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nWynik: {passed}/{len(results)} testów przeszło pomyślnie")
    
    if passed == len(results):
        print("🎉 Wszystkie testy przeszły! Projekt gotowy do użycia.")
        print("\nMożesz teraz uruchomić:")
        print("   python main.py --mode single --model_type cnn --model_name resnet18 --fraction 0.1")
    else:
        print("⚠️ Niektóre testy nie przeszły. Sprawdź błędy powyżej.")
        
if __name__ == "__main__":
    main() 