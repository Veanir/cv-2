"""
Skrypt do analizy dystrybucji klas w zbiorze danych HAM10000.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Poprawki dla wyświetlania na serwerach bez GUI
import matplotlib
matplotlib.use('Agg')

def analyze_dataset_distribution(data_dir: str = 'data', results_dir: str = 'results'):
    """
    Analizuje i wizualizuje rozkład klas w pliku ze splitami danych.
    """
    splits_path = os.path.join(data_dir, "ham10000_splits.csv")
    
    if not os.path.exists(splits_path):
        print(f"❌ Plik podziału '{splits_path}' nie został znaleziony.")
        print("   Proszę najpierw uruchomić skrypt: python scripts/prepare_dataset.py")
        return

    print(f"📖 Wczytywanie pliku podziału danych z: {splits_path}")
    df = pd.read_csv(splits_path)

    # Analiza ogólna
    print("\n" + "="*50)
    print("📊 Ogólna dystrybucja klas w całym zbiorze danych")
    print("="*50)
    total_distribution = df['dx'].value_counts()
    total_percentage = df['dx'].value_counts(normalize=True) * 100
    
    summary_df = pd.DataFrame({
        'Liczba próbek': total_distribution,
        'Procentowo (%)': total_percentage.round(2)
    })
    print(summary_df)
    print(f"\nCałkowita liczba próbek: {len(df)}")


    # Analiza dla każdego podzbioru
    print("\n" + "="*50)
    print("📊 Dystrybucja klas w poszczególnych podzbiorach")
    print("="*50)
    split_distribution = df.groupby('split')['dx'].value_counts(normalize=True).unstack(level=1) * 100
    print(split_distribution.round(2))

    # Wizualizacja dystrybucji
    print("\n🎨 Generowanie wizualizacji dystrybucji klas...")
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x='dx', order=summary_df.index, palette='viridis')
    plt.title('Dystrybucja klas w zbiorze danych HAM10000', fontsize=16)
    plt.xlabel('Klasa (typ zmiany skórnej)', fontsize=12)
    plt.ylabel('Liczba próbek', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Dodanie etykiet z liczbą próbek na słupkach
    for i, count in enumerate(summary_df['Liczba próbek']):
        plt.text(i, count + 50, str(count), ha='center', va='bottom')

    plt.tight_layout()
    
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'dataset_class_distribution.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"✅ Wizualizacja została zapisana w: {save_path}")

if __name__ == '__main__':
    analyze_dataset_distribution() 