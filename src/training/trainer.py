"""
Moduł do treningu modeli Vision Transformer vs CNN
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

class ModelTrainer:
    """Klasa do treningu modeli"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: torch.device,
                 experiment_name: str = "medical_classification"):
        """
        Args:
            model: Model do treningu
            train_loader: DataLoader dla danych treningowych
            val_loader: DataLoader dla danych walidacyjnych
            config: Konfiguracja treningu
            device: Urządzenie (cuda/cpu)
            experiment_name: Nazwa eksperymentu
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Historia treningu
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'f1': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'f1': []
        }
        
        # Setup optymalizatora
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = self._setup_criterion()
        
        # Setup logowania
        self.use_wandb = config.get('log_wandb', False)
        if self.use_wandb and wandb.run is not None:
            # Obserwuj model, jeśli run jest aktywny
            wandb.watch(self.model)
            
        # Early stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = config.get('patience', 15)
        self.min_delta = config.get('min_delta', 0.001)
        
        # Checkpointing
        self.checkpoint_dir = os.path.join("checkpoints", experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Konfiguruje optymalizator"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Nieznany optymalizator: {optimizer_name}")
            
    def _setup_scheduler(self):
        """Konfiguruje scheduler learning rate"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', patience=10, factor=0.5)
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Nieznany scheduler: {scheduler_name}")
            
    def _setup_criterion(self) -> nn.Module:
        """
        Konfiguruje funkcję straty. Jeśli w konfiguracji jest włączone ważenie,
        dynamicznie oblicza wagi na podstawie aktualnego `train_loader`.
        """
        # Sprawdzenie, czy stosować ważenie klas
        if self.config.get('use_class_weighting', True):
            print("⚖️ Obliczanie wag klas na podstawie danych treningowych...")
            
            # Pobierz liczbę klas z modelu
            num_classes = self.model.num_classes
            
            # Iteruj po loaderze, aby zliczyć klasy
            class_counts = torch.zeros(num_classes, dtype=torch.float)
            for _, labels, _ in self.train_loader:
                for label in labels:
                    class_counts[label] += 1
            
            # Unikaj dzielenia przez zero
            total_samples = class_counts.sum()
            if total_samples == 0:
                 print("⚠️ Brak danych w train_loader, używam standardowej funkcji straty.")
                 return nn.CrossEntropyLoss()

            # Oblicz wagi (inverse frequency)
            class_weights = total_samples / (num_classes * class_counts)
            
            # Obsługa klas, które mogą nie wystąpić w małych próbkach
            class_weights[class_counts == 0] = 1.0 
            
            print(f"   Obliczone wagi: {[f'{w:.2f}' for w in class_weights]}")
            
            return nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            print("ℹ️ Stosowanie standardowej funkcji straty (bez ważenia).")
            return nn.CrossEntropyLoss()
            
    def train_epoch(self) -> Dict[str, float]:
        """Trenuje jeden epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statystyki
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': running_loss / (batch_idx + 1)
            })
            
        # Oblicz metryki
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_predictions)
        epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1
        }
        
    def validate_epoch(self) -> Dict[str, float]:
        """Waliduje jeden epoch"""
        # Sprawdź czy val_loader ma próbki
        if len(self.val_loader) == 0:
            print("⚠️ Brak próbek walidacyjnych - pomijam walidację")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1': 0.0,
                'predictions': [],
                'labels': []
            }
            
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch_idx, (images, labels, _) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statystyki
                running_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': running_loss / (batch_idx + 1)
                })
                
        # Oblicz metryki
        epoch_loss = running_loss / len(self.val_loader)
        
        if len(all_labels) > 0:
            epoch_acc = accuracy_score(all_labels, all_predictions)
            epoch_f1 = f1_score(all_labels, all_predictions, average='weighted')
        else:
            epoch_acc = 0.0
            epoch_f1 = 0.0
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Zapisuje checkpoint modelu"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        # Zapisz ostatni checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'last_checkpoint.pth'))
        
        # Zapisz najlepszy model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"Zapisano najlepszy model (epoch {epoch})")

            # Loguj najlepszy model do WandB jako artefakt
            if self.use_wandb and wandb.run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"model-{self.experiment_name}",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "val_loss": self.best_val_loss,
                            "val_acc": self.best_val_acc,
                            "model_class": self.model.__class__.__name__
                        }
                    )
                    artifact.add_file(self.best_model_path)
                    wandb.log_artifact(artifact, aliases=["best", f"epoch_{epoch}"])
                    print("🧹 Zalogowano najlepszy model do WandB.")
                except Exception as e:
                    print(f"⚠️ Nie udało się zalogować modelu do WandB: {e}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Ładuje checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        return checkpoint['epoch']
        
    def early_stopping_check(self, val_loss: float) -> Tuple[bool, bool]:
        """
        Sprawdza warunki early stopping na podstawie val_loss.
        
        Zwraca:
            Tuple[bool, bool]: (czy należy zatrzymać trening, czy to jest najlepszy model)
        """
        is_best = False
        if val_loss < self.best_val_loss - self.min_delta:
            # Poprawa, zapisz model i zresetuj cierpliwość
            self.best_val_loss = val_loss
            self.patience_counter = 0
            is_best = True
            print(f"✅ Poprawa walidacji! Nowa najlepsza strata: {self.best_val_loss:.4f}")
        else:
            # Brak poprawy, zwiększ licznik cierpliwości
            self.patience_counter += 1
            print(f"⚠️ Brak poprawy. Cierpliwość: {self.patience_counter}/{self.patience}")

        should_stop = self.patience_counter >= self.patience
        if should_stop:
            print(f"⏳ Early stopping! Osiągnięto limit cierpliwości.")

        return should_stop, is_best
        
    def train(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Główna funkcja treningu"""
        if epochs is None:
            epochs = self.config.get('epochs', 100)
            
        print(f"Rozpoczynam trening na {epochs} epok...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Trening
            train_metrics = self.train_epoch()
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['f1'].append(train_metrics['f1'])
            
            # Walidacja
            val_metrics = self.validate_epoch()
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['accuracy'].append(val_metrics['accuracy'])
            self.val_history['f1'].append(val_metrics['f1'])
            
            # Wyświetl metryki
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            # Log do wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
            # Early stopping i zapis checkpointa
            should_stop, is_best = self.early_stopping_check(val_metrics['loss'])
            self.save_checkpoint(epoch + 1, is_best=is_best)

            if should_stop:
                break
                
        total_time = time.time() - start_time
        print(f"\nTrening zakończony w {total_time:.2f} sekund")
        print(f"Najlepsza walidacyjna dokładność: {self.best_val_acc:.4f}")
        print(f"Najlepszy model zapisany w: {self.best_model_path}")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }

    def get_best_model(self) -> nn.Module:
        """Ładuje i zwraca najlepszy model z checkpointa"""
        print(f"Ładowanie najlepszego modelu z: {self.best_model_path}")
        if not os.path.exists(self.best_model_path):
            print("⚠️ Nie znaleziono najlepszego modelu, zwracam aktualny.")
            return self.model
            
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model


if __name__ == "__main__":
    # Test trainera
    print("Testowanie ModelTrainer...")
    
    # Przykładowa konfiguracja
    config = {
        'epochs': 2,
        'learning_rate': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'patience': 5,
        'log_wandb': False
    }
    
    # Użyj CPU dla testu
    device = torch.device('cpu')
    
    # Przykładowy model i dane
    from src.models.cnn_models import MedicalCNN
    model = MedicalCNN(model_name="resnet18", num_classes=7)
    
    # Przykładowe dane (zastąp prawdziwymi)
    from torch.utils.data import TensorDataset
    X = torch.randn(32, 3, 224, 224)
    y = torch.randint(0, 7, (32,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=8)
    val_loader = DataLoader(dataset, batch_size=8)
    
    trainer = ModelTrainer(model, train_loader, val_loader, config, device, "test")
    history = trainer.train()

    # Test ewaluacji (teraz poza trenerem)
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(
        model=trainer.get_best_model(),
        device=device,
        class_names=[f'Class_{i}' for i in range(7)]
    )
    results = evaluator.evaluate(val_loader)
    print(f"Wyniki ewaluacji testowej: Accuracy={results['accuracy']:.4f}")
    
    print("Test zakończony pomyślnie!") 