"""
Moduł do wizualizacji wyników i interpretabilności modeli.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Poprawki dla wyświetlania na serwerach bez GUI
import matplotlib
matplotlib.use('Agg')

from .interpretability import overlay_heatmap

class ResultVisualizer:
    """Klasa do tworzenia wizualizacji wyników eksperymentów"""
    
    def __init__(self, results_dir: str, class_names: List[str]):
        """
        Args:
            results_dir: Folder do zapisu wizualizacji
            class_names: Lista nazw klas
        """
        self.results_dir = results_dir
        self.class_names = class_names
        os.makedirs(self.results_dir, exist_ok=True)
        
    def plot_learning_curves(self, history: Dict[str, Any]):
        """Rysuje krzywe uczenia (loss i accuracy)"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Krzywe straty
        train_loss = history['train_history']['loss']
        val_loss = history['val_history']['loss']
        ax1.plot(train_loss, label='Train Loss')
        ax1.plot(val_loss, label='Validation Loss')
        ax1.set_title('Krzywa Straty (Loss)')
        ax1.set_xlabel('Epoka')
        ax1.set_ylabel('Strata')
        ax1.legend()
        ax1.grid(True)
        
        # Krzywe dokładności
        train_acc = history['train_history']['accuracy']
        val_acc = history['val_history']['accuracy']
        ax2.plot(train_acc, label='Train Accuracy')
        ax2.plot(val_acc, label='Validation Accuracy')
        ax2.set_title('Krzywa Dokładności (Accuracy)')
        ax2.set_xlabel('Epoka')
        ax2.set_ylabel('Dokładność')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'learning_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Krzywe uczenia zapisane w: {save_path}")
        
    def plot_confusion_matrix(self, cm: np.ndarray, normalize: bool = False):
        """Rysuje macierz pomyłek"""
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Znormalizowana Macierz Pomyłek'
        else:
            fmt = 'd'
            title = 'Macierz Pomyłek'
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Macierz pomyłek zapisana w: {save_path}")

    def save_classification_report(self, report: Dict[str, Any]):
        """Zapisuje raport klasyfikacji do pliku CSV"""
        report_df = pd.DataFrame(report).transpose()
        save_path = os.path.join(self.results_dir, 'classification_report.csv')
        report_df.to_csv(save_path, index=True)
        print(f"✅ Raport klasyfikacji zapisany w: {save_path}")

    def plot_predictions(self, 
                         images: np.ndarray, 
                         labels: np.ndarray, 
                         predictions: np.ndarray,
                         max_samples: int = 16):
        """Wizualizuje przykładowe predykcje modelu"""
        
        num_samples = min(max_samples, len(images))
        if num_samples == 0:
            return

        cols = 4
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 4 * rows))
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = np.transpose(images[i], (1, 2, 0)) # z (C,H,W) na (H,W,C)
            img = np.clip(img, 0, 1) # Normalizacja do [0,1]
            
            true_label = self.class_names[labels[i]]
            pred_label = self.class_names[predictions[i]]
            
            axes[i].imshow(img)
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
            axes[i].axis('off')
            
        for j in range(num_samples, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'sample_predictions.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Przykładowe predykcje zapisane w: {save_path}")

    def plot_interpretability_maps(self, 
                                   images: np.ndarray, 
                                   maps: List[np.ndarray],
                                   max_samples: int = 16):
        """Wizualizuje mapy interpretabilności (Grad-CAM/Attention) na obrazach"""
        num_samples = min(max_samples, len(images), len(maps))
        if num_samples == 0:
            print("⚠️ Brak map interpretabilności do wizualizacji.")
            return

        cols = 4
        rows = int(np.ceil(num_samples / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()
        
        for i in range(num_samples):
            img = images[i]
            heatmap = maps[i]
            
            overlaid_img = overlay_heatmap(heatmap, img, alpha=0.6)
            
            axes[i].imshow(overlaid_img)
            axes[i].set_title(f"Przykład {i+1}")
            axes[i].axis('off')

        for j in range(num_samples, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'interpretability_maps.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Mapy interpretabilności zapisane w: {save_path}")

    def run_all_visualizations(self, 
                               history: Dict[str, Any], 
                               eval_results: Dict[str, Any]):
        """Uruchamia wszystkie wizualizacje"""
        print("\n🎨 Generowanie wizualizacji...")
        
        # Krzywe uczenia
        if history:
            self.plot_learning_curves(history)
        
        # Macierz pomyłek
        if 'confusion_matrix' in eval_results:
            self.plot_confusion_matrix(eval_results['confusion_matrix'])
        
        # Raport klasyfikacji
        if 'classification_report' in eval_results:
            self.save_classification_report(eval_results['classification_report'])
            
        # Przykładowe predykcje
        if 'visualization_samples' in eval_results:
            samples = eval_results['visualization_samples']
            self.plot_predictions(
                images=samples['images'],
                labels=samples['labels'],
                predictions=samples['predictions']
            )
            
            # Mapy interpretabilności
            if 'interpretability_maps' in samples and samples['interpretability_maps']:
                self.plot_interpretability_maps(
                    images=samples['images'],
                    maps=samples['interpretability_maps']
                )
        
        print("✅ Wszystkie wizualizacje zostały wygenerowane.")

if __name__ == '__main__':
    # Przykładowe użycie
    print("Testowanie ResultVisualizer...")
    
    # Tworzenie sztucznych danych
    results_dir = "results/test_visualizer"
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    visualizer = ResultVisualizer(results_dir, class_names)
    
    # Przykładowa historia
    history = {
        'train_history': {'loss': [0.5, 0.3, 0.2], 'accuracy': [0.8, 0.85, 0.9]},
        'val_history': {'loss': [0.6, 0.4, 0.3], 'accuracy': [0.75, 0.82, 0.88]}
    }
    
    # Przykładowe wyniki ewaluacji
    eval_results = {
        'confusion_matrix': np.random.randint(0, 50, size=(7, 7)),
        'classification_report': {
            'akiec': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75, 'support': 10},
            'bcc': {'precision': 0.9, 'recall': 0.85, 'f1-score': 0.87, 'support': 20}
        },
        'visualization_samples': {
            'images': np.random.rand(16, 3, 224, 224),
            'labels': np.random.randint(0, 7, size=16),
            'predictions': np.random.randint(0, 7, size=16),
            'interpretability_maps': [np.random.rand(14, 14) for _ in range(16)]
        }
    }
    
    visualizer.run_all_visualizations(history, eval_results)
    print("\nTest zakończony. Sprawdź folder 'results/test_visualizer'.") 