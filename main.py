"""
Główny skrypt do uruchamiania eksperymentów Vision Transformer vs CNN
"""
import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime
import json

# Import konfiguracji
from config import (
    data_config, model_config, training_config, experiment_config
)

# Import modułów
from src.data.dataset import create_dataloaders
from src.models.vision_transformer import create_vit_model
from src.models.cnn_models import create_cnn_model
from src.training.trainer import ModelTrainer
# from src.evaluation.evaluator import ModelEvaluator  # TODO: Implement later
# from src.evaluation.visualizer import ResultVisualizer  # TODO: Implement later

def set_seed(seed: int = 42):
    """Ustawia seed dla reprodukowalności"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
def get_device():
    """Zwraca dostępne urządzenie"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Używam CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Używam CPU")
    return device

def create_experiment_config(model_type: str, 
                           model_name: str,
                           dataset_fraction: float = 1.0,
                           fine_tune: bool = False):
    """Tworzy konfigurację eksperymentu"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{model_type}_{model_name.replace('/', '_')}_{timestamp}"
    
    if dataset_fraction < 1.0:
        experiment_name += f"_frac_{dataset_fraction}"
    if fine_tune:
        experiment_name += "_finetune"
        
    config = {
        'model_type': model_type,
        'model_name': model_name,
        'dataset_fraction': dataset_fraction,
        'fine_tune': fine_tune,
        'experiment_name': experiment_name,
        'timestamp': timestamp
    }
    
    return config

def run_single_experiment(model_type: str,
                         model_name: str, 
                         dataset_fraction: float = 1.0,
                         fine_tune: bool = False):
    """Uruchamia pojedynczy eksperyment"""
    
    print(f"\n{'='*60}")
    print(f"EKSPERYMENT: {model_type.upper()} - {model_name}")
    print(f"Dataset fraction: {dataset_fraction}")
    print(f"Fine-tuning: {fine_tune}")
    print(f"{'='*60}\n")
    
    # Ustawienia
    set_seed(42)
    device = get_device()
    
    # Konfiguracja eksperymentu
    exp_config = create_experiment_config(
        model_type, model_name, dataset_fraction, fine_tune
    )
    
    # Załaduj dane
    print("Ładowanie danych...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_config.data_dir,
        batch_size=data_config.batch_size,
        img_size=data_config.img_size,
        num_workers=data_config.num_workers
    )
    
    # Ogranicz rozmiar datasetu jeśli wymagane
    if dataset_fraction < 1.0:
        print(f"Ograniczam dataset do {dataset_fraction*100}% próbek...")
        train_loader = limit_dataloader(train_loader, dataset_fraction)
        val_loader = limit_dataloader(val_loader, dataset_fraction)
    
    print(f"Rozmiary zbiorów:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val: {len(val_loader.dataset)}")
    print(f"  Test: {len(test_loader.dataset)}")
    
    # Utwórz model
    print(f"\nTworzenie modelu {model_type}...")
    if model_type == 'vit':
        model_config_dict = {
            'type': 'single',
            'vit_model_name': model_name,
            'num_classes': model_config.num_classes,
            'vit_pretrained': model_config.vit_pretrained,
            'freeze_backbone': model_config.freeze_backbone and not fine_tune,
            'dropout_rate': model_config.dropout_rate
        }
        model = create_vit_model(model_config_dict)
    elif model_type == 'cnn':
        model_config_dict = {
            'type': 'single',
            'cnn_model_name': model_name,
            'num_classes': model_config.num_classes,
            'cnn_pretrained': model_config.cnn_pretrained,
            'freeze_backbone': model_config.freeze_backbone and not fine_tune,
            'dropout_rate': model_config.dropout_rate
        }
        model = create_cnn_model(model_config_dict)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")
        
    # Informacje o modelu
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parametry modelu:")
    print(f"  Łącznie: {total_params:,}")
    print(f"  Trenowalne: {trainable_params:,}")
    print(f"  Zamrożone: {total_params - trainable_params:,}")
    
    # Konfiguracja treningu
    training_config_dict = {
        'epochs': training_config.epochs,
        'learning_rate': training_config.learning_rate,
        'weight_decay': training_config.weight_decay,
        'optimizer': training_config.optimizer,
        'scheduler': training_config.scheduler,
        'patience': training_config.patience,
        'min_delta': training_config.min_delta,
        'log_wandb': experiment_config.log_wandb
    }
    
    # Trening
    print("\nRozpoczynanie treningu...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config_dict,
        device=device,
        experiment_name=exp_config['experiment_name']
    )
    
    history = trainer.train()
    
    # Ewaluacja
    print("\nEwaluacja modelu...")
    test_results = trainer.evaluate(test_loader)
    
    # Zapisz wyniki
    results_dir = os.path.join(experiment_config.output_dir, exp_config['experiment_name'])
    os.makedirs(results_dir, exist_ok=True)
    
    # Zapisz konfigurację
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        config_to_save = {
            **exp_config,
            'model_config': model_config_dict,
            'training_config': training_config_dict,
            'data_config': {
                'batch_size': data_config.batch_size,
                'img_size': data_config.img_size,
                'dataset_name': data_config.dataset_name
            }
        }
        json.dump(config_to_save, f, indent=2)
    
    # Zapisz wyniki
    results_to_save = {
        'history': history,
        'test_results': {
            'accuracy': test_results['accuracy'],
            'f1_weighted': test_results['f1_weighted'],
            'classification_report': test_results['classification_report']
        },
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_type': model_type,
            'model_name': model_name
        }
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"\nWyniki zapisane w: {results_dir}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test F1: {test_results['f1_weighted']:.4f}")
    
    return results_to_save

def limit_dataloader(dataloader, fraction):
    """Ogranicza rozmiar DataLoader'a"""
    dataset = dataloader.dataset
    total_size = len(dataset)
    limited_size = int(total_size * fraction)
    
    indices = list(range(total_size))[:limited_size]
    limited_dataset = torch.utils.data.Subset(dataset, indices)
    
    return torch.utils.data.DataLoader(
        limited_dataset,
        batch_size=dataloader.batch_size,
        shuffle=dataloader.drop_last,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory
    )

def run_comparison_study():
    """Uruchamia pełne porównanie ViT vs CNN"""
    
    print("ROZPOCZYNAM PEŁNE BADANIE PORÓWNAWCZE: VISION TRANSFORMER VS CNN")
    print("=" * 80)
    
    # Modele do testowania
    models_to_test = [
        # Vision Transformers
        ('vit', 'google/vit-base-patch16-224'),
        # CNNs
        ('cnn', 'resnet50'),
        ('cnn', 'efficientnet_b0'),
    ]
    
    # Frakcje datasetu do testowania
    dataset_fractions = [0.1, 0.25, 0.5, 1.0]
    
    all_results = []
    
    for fraction in dataset_fractions:
        print(f"\n\nTESTOWANIE Z {fraction*100}% DANYCH")
        print("-" * 40)
        
        for model_type, model_name in models_to_test:
            try:
                results = run_single_experiment(
                    model_type=model_type,
                    model_name=model_name,
                    dataset_fraction=fraction,
                    fine_tune=False
                )
                results['dataset_fraction'] = fraction
                all_results.append(results)
                
            except Exception as e:
                print(f"BŁĄD w eksperymencie {model_type}-{model_name}: {e}")
                continue
    
    # Zapisz podsumowanie
    summary_results = []
    for result in all_results:
        summary_results.append({
            'model_type': result['model_info']['model_type'],
            'model_name': result['model_info']['model_name'],
            'dataset_fraction': result['dataset_fraction'],
            'test_accuracy': result['test_results']['accuracy'],
            'test_f1': result['test_results']['f1_weighted'],
            'total_params': result['model_info']['total_params'],
            'trainable_params': result['model_info']['trainable_params']
        })
    
    # Zapisz podsumowanie
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(experiment_config.output_dir, f'comparison_summary_{timestamp}.json')
    
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print(f"\n\nPODSUMOWANIE ZAPISANE W: {summary_path}")
    
    # Wyświetl podsumowanie
    print("\nPODSUMOWANIE WYNIKÓW:")
    print("-" * 80)
    for result in summary_results:
        print(f"{result['model_type']:<4} {result['model_name']:<25} "
              f"Frac: {result['dataset_fraction']:<4} "
              f"Acc: {result['test_accuracy']:.4f} "
              f"F1: {result['test_f1']:.4f} "
              f"Params: {result['total_params']:>10,}")

def main():
    parser = argparse.ArgumentParser(description='Vision Transformer vs CNN - Klasyfikacja medyczna')
    parser.add_argument('--mode', choices=['single', 'comparison'], default='comparison',
                       help='Tryb uruchomienia')
    parser.add_argument('--model_type', choices=['vit', 'cnn'], 
                       help='Typ modelu (dla trybu single)')
    parser.add_argument('--model_name', type=str,
                       help='Nazwa modelu (dla trybu single)')
    parser.add_argument('--fraction', type=float, default=1.0,
                       help='Frakcja datasetu (0.1-1.0)')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Czy użyć fine-tuningu')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.model_type or not args.model_name:
            print("W trybie single wymagane są --model_type i --model_name")
            return
            
        run_single_experiment(
            model_type=args.model_type,
            model_name=args.model_name,
            dataset_fraction=args.fraction,
            fine_tune=args.fine_tune
        )
    else:
        run_comparison_study()

if __name__ == "__main__":
    main() 