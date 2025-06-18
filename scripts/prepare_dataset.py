"""
Skrypt do jednorazowego przygotowania i podziału datasetu HAM10000.

Tworzy plik `ham10000_splits.csv` z podziałem na zbiory
treningowy, walidacyjny i testowy, co zapewnia spójność
i przyspiesza ładowanie danych w głównym skrypcie.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

def find_image_paths(data_dir: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Znajduje i mapuje ścieżki do obrazów dla każdego image_id."""
    print("🗺️ Mapowanie ścieżek do obrazów...")
    
    path_map = {}
    image_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2']
    
    for folder in image_dirs:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Ostrzeżenie: Folder '{folder_path}' nie istnieje.")
            continue
        
        for image_file in tqdm(os.listdir(folder_path), desc=f"Przeszukiwanie {folder}"):
            if image_file.endswith('.jpg'):
                image_id = os.path.splitext(image_file)[0]
                path_map[image_id] = os.path.join(folder, image_file)
                
    metadata['path'] = metadata['image_id'].map(path_map)
    
    missing_paths = metadata['path'].isnull().sum()
    if missing_paths > 0:
        print(f"❌ Nie znaleziono {missing_paths} obrazów! Upewnij się, że dataset jest kompletny.")
    else:
        print("✅ Pomyślnie zmapowano wszystkie obrazy.")
        
    return metadata.dropna(subset=['path'])

def create_splits(data_dir: str = 'data', 
                  val_size: float = 0.15, 
                  test_size: float = 0.15, 
                  random_state: int = 42):
    """
    Tworzy stratyfikowany podział danych i zapisuje go do pliku CSV.
    """
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    output_path = os.path.join(data_dir, 'ham10000_splits.csv')
    
    if not os.path.exists(metadata_path):
        print(f"❌ Nie znaleziono pliku metadanych: {metadata_path}")
        print("   Upewnij się, że dataset HAM10000 został pobrany i umieszczony w folderze 'data'.")
        return

    print("📖 Wczytywanie metadanych...")
    metadata = pd.read_csv(metadata_path)
    
    # Krok 1: Mapowanie ścieżek
    metadata = find_image_paths(data_dir, metadata)
    
    # Przygotowanie do podziału
    labels = metadata['dx'] # 'dx' to kolumna z etykietami klas
    
    # Krok 2: Podział na train i (val + test)
    print(f"🔪 Dzielenie danych (test={test_size}, val={val_size})...")
    
    # Najpierw oddzielamy zbiór testowy
    splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_test.split(metadata, labels))
    
    # Stworzenie ramek danych
    train_val_set = metadata.iloc[train_val_idx]
    test_set = metadata.iloc[test_idx]
    
    # Przypisanie splitu
    metadata['split'] = 'test'
    metadata.iloc[train_val_idx, metadata.columns.get_loc('split')] = 'train' # Domyślnie train

    # Krok 3: Podział (train + val) na train i val
    val_size_adjusted = val_size / (1.0 - test_size)
    labels_train_val = train_val_set['dx']
    
    splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, val_idx = next(splitter_val.split(train_val_set, labels_train_val))
    
    # Indeksy w oryginalnej ramce danych
    original_train_idx = train_val_set.iloc[train_idx].index
    original_val_idx = train_val_set.iloc[val_idx].index
    
    # Aktualizacja splitu
    metadata.loc[original_train_idx, 'split'] = 'train'
    metadata.loc[original_val_idx, 'split'] = 'val'
    
    # Zapis do pliku
    final_df = metadata[['image_id', 'dx', 'split', 'path']]
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Podział danych zapisany w: {output_path}")
    print("\n📊 Rozkład próbek w podziałach:")
    print(final_df['split'].value_counts())

if __name__ == '__main__':
    create_splits() 