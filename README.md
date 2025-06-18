# Vision Transformer vs CNN - Klasyfikacja medyczna

Projekt badawczy porównujący efektywność Vision Transformers i Convolutional Neural Networks w klasyfikacji dermatologicznej.

## 📋 Hipoteza badawcza

**"Vision Transformers osiągają lepszą dokładność niż tradycyjne CNN w klasyfikacji zdjęć dermatologicznych, szczególnie przy ograniczonych danych treningowych"**

## 🚀 Szybki start

### Opcja 1: Docker (zalecana) 🐳

```bash
# 1. Zbuduj obraz Docker
docker-compose build

# 2. Uruchom testy
docker-compose run --rm test

# 3. Pierwszy eksperyment
docker-compose run --rm experiment
```

**Lub użyj interaktywnych skryptów:**
- **Linux/Mac**: `bash scripts/docker_run.sh`
- **Windows**: `scripts\docker_run.bat`

### Opcja 2: Instalacja lokalna

```bash
# 1. Instalacja zależności
pip install -r requirements.txt

# 2. Test konfiguracji
python test_setup.py

# 3. Pierwszy eksperyment
python main.py --mode single --model_type cnn --model_name resnet18 --fraction 0.1
```

## 📁 Struktura projektu

```
project_2/
├── src/
│   ├── data/
│   │   ├── dataset.py          # HAM10000 dataset loader
│   │   └── __init__.py
│   ├── models/
│   │   ├── vision_transformer.py    # ViT implementation
│   │   ├── cnn_models.py            # CNN implementations
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── __init__.py
│   └── evaluation/
│       └── __init__.py
├── data/                       # Dane (tworzone automatycznie)
├── results/                    # Wyniki eksperymentów
├── checkpoints/                # Zapisane modele
├── logs/                       # Logi treningów
├── config.py                   # Konfiguracja projektu
├── main.py                     # Główny skrypt
├── test_setup.py               # Test konfiguracji
├── requirements.txt            # Zależności
└── README.md                   # Ten plik
```

## 🔧 Konfiguracja

Główne ustawienia w `config.py`:

- **Dataset**: HAM10000 (7 klas dermatologicznych)
- **Modele**: ViT-Base, ResNet50, EfficientNet-B0
- **Batch size**: 32
- **Learning rate**: 1e-4
- **Epochs**: 100 (z early stopping)

## 🧪 Dostępne eksperymenty

### Z Docker (zalecane) 🐳

```bash
# Szybkie testy
docker-compose run --rm experiment  # CNN ResNet18, 10% danych
docker-compose run --rm test        # Testy konfiguracji

# Własne eksperymenty
docker-compose run --rm experiment python main.py \
  --mode single --model_type vit --model_name google/vit-base-patch16-224 --fraction 0.1

# Pełne porównanie (długie!)
docker-compose run --rm comparison

# Interaktywny terminal
docker-compose run --rm -it vit-cnn-research bash
```

### Eksperymenty lokalne

```bash
# CNN ResNet50
python main.py --mode single --model_type cnn --model_name resnet50

# Vision Transformer
python main.py --mode single --model_type vit --model_name google/vit-base-patch16-224

# Z ograniczonym zbiorem danych
python main.py --mode single --model_type cnn --model_name resnet50 --fraction 0.25

# Z fine-tuningiem
python main.py --mode single --model_type vit --model_name google/vit-base-patch16-224 --fine_tune

# Pełne porównanie
python main.py --mode comparison
```

## 📊 Dostępne modele

### Vision Transformers
- `google/vit-base-patch16-224` - ViT Base
- `google/vit-large-patch16-224` - ViT Large

### CNN
- `resnet18`, `resnet50`, `resnet101` - ResNet family
- `efficientnet_b0`, `efficientnet_b1` - EfficientNet family
- `densenet121` - DenseNet

## 🎯 Eksperymenty do hipotezy

Projekt automatycznie testuje:

1. **Wpływ rozmiaru datasetu** (10%, 25%, 50%, 100% danych)
2. **Porównanie architektur** (ViT vs różne CNN)
3. **Transfer learning** (pretrenowane vs from scratch)
4. **Fine-tuning strategies** (zamrożone vs odmarożone warstwy)

## 📈 Wyniki

Wyniki są zapisywane w folderze `results/` w formacie:

```
results/
├── vit_google_vit-base-patch16-224_20241215_143022/
│   ├── config.json
│   ├── results.json
│   └── best_model.pth
└── comparison_summary_20241215_150000.json
```

### Interpretacja wyników

- `accuracy` - Dokładność klasyfikacji
- `f1_weighted` - F1-score ważony
- `classification_report` - Szczegółowy raport dla każdej klasy
- `confusion_matrix` - Macierz pomyłek

## 🔬 Analiza rezultatów

Po uruchomieniu eksperymentów możesz analizować:

```python
import json

# Załaduj wyniki
with open('results/comparison_summary_TIMESTAMP.json', 'r') as f:
    results = json.load(f)

# Znajdź najlepszy model
best_model = max(results, key=lambda x: x['test_accuracy'])
print(f"Najlepszy model: {best_model['model_type']} - {best_model['model_name']}")
print(f"Dokładność: {best_model['test_accuracy']:.4f}")
```

## 🐳 Docker - Szczegóły

### Dostępne serwisy

```bash
# Testy konfiguracji
docker-compose run --rm test

# Pojedynczy eksperyment
docker-compose run --rm experiment 

# Pełne porównanie
docker-compose run --rm comparison

# Interaktywny terminal
docker-compose run --rm -it vit-cnn-research bash
```

### GPU Support w Docker

Aby używać GPU, odkomentuj sekcję w `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

I zainstaluj [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

### Volumes

Docker automatycznie mountuje:
- `./data` → `/app/data` (dane)
- `./results` → `/app/results` (wyniki)
- `./logs` → `/app/logs` (logi)
- `./checkpoints` → `/app/checkpoints` (modele)

## 🐛 Troubleshooting

### Problem: Docker nie buduje się
```
ERROR: failed to solve: failed to compute cache key
```
**Rozwiązanie**: `docker system prune -a` i spróbuj ponownie.

### Problem: Brak CUDA
```
⚠️ CUDA niedostępna - używam CPU
```
**Rozwiązanie**: To normalne. Dla testów CPU wystarczy. Dla GPU zobacz sekcję "GPU Support".

### Problem: Błąd internetu przy ViT
```
⚠️ ViT niedostępny (prawdopodobnie brak internetu)
```
**Rozwiązanie**: ViT wymaga pobrania z HuggingFace. Sprawdź połączenie internetowe.

### Problem: Błąd pamięci
```
RuntimeError: CUDA out of memory
```
**Rozwiązanie**: Zmniejsz `batch_size` w `config.py` lub użyj `--fraction 0.1`.

### Problem: Kontenery nie usuwają się
```bash
docker-compose down
docker system prune -f
```

## 📚 Dataset

Projekt używa HAM10000 - dataset do klasyfikacji zmian skórnych:

- **Klasy**: 7 (akiec, bcc, bkl, df, mel, nv, vasc)
- **Obrazy**: ~10,000 zdjęć dermatologicznych
- **Format**: JPEG, normalizowane do 224x224

### Pobieranie prawdziwych danych

Dla najlepszych wyników pobierz HAM10000 z:
- [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ISIC Archive](https://challenge.isic-archive.com/data/)

Umieść w folderze `data/`:
```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

## 🎓 Cele badawcze

1. **Porównanie dokładności** ViT vs CNN
2. **Analiza wpływu rozmiaru datasetu** na performance
3. **Badanie transfer learning** i fine-tuning strategies
4. **Interpretabilność** - attention maps vs feature maps
5. **Efektywność obliczeniowa** - czas treningu, liczba parametrów

## 📝 Raportowanie

Projekt generuje wszystkie dane potrzebne do raportu:

- **Accuracy metrics** dla różnych modeli
- **Confusion matrices** 
- **Learning curves**
- **Parameter counts**
- **Training times**

## 🤝 Kontakt

W przypadku problemów sprawdź:
1. `test_setup.py` - diagnostyka
2. Logi w folderze `logs/`
3. Dokumentację PyTorch/Transformers

---

**Powodzenia w badaniach! 🔬🎯** 