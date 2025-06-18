"""
Moduł do ewaluacji modeli
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

from .interpretability import GradCAM, get_cnn_target_layer, get_vit_attention_map

class ModelEvaluator:
    """Klasa do kompleksowej ewaluacji modeli"""
    
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device, 
                 class_names: List[str]):
        """
        Args:
            model: Wytrenowany model do ewaluacji
            device: Urządzenie (cuda/cpu)
            class_names: Lista nazw klas
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model_type = 'vit' if 'vit' in model.__class__.__name__.lower() else 'cnn'
        
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Ewaluuje model na danym zbiorze danych (np. testowym)

        Args:
            test_loader: DataLoader ze zbiorem testowym

        Returns:
            Słownik z kompleksowymi wynikami ewaluacji
        """
        print("📊 Rozpoczynam ewaluację modelu...")
        
        self.model.eval()
        all_predictions, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="Ewaluacja"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        report = classification_report(
            all_labels, all_predictions, target_names=self.class_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Zbierz próbki i wygeneruj mapy
        tensors_for_maps, numpy_for_plots = self._get_samples_for_visualization(test_loader)
        interpretability_maps = self._generate_interpretability_maps(
            tensors_for_maps['images'], tensors_for_maps['labels']
        )
        
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probs,
            'visualization_samples': {
                'images': numpy_for_plots['images'],
                'labels': numpy_for_plots['labels'],
                'predictions': numpy_for_plots['predictions'],
                'interpretability_maps': interpretability_maps
            }
        }
        
        print(f"✅ Ewaluacja zakończona. Dokładność: {accuracy:.4f}, F1-score: {f1:.4f}")
        return results

    def _get_samples_for_visualization(self, dataloader: DataLoader, num_samples: int = 16):
        """
        Pobiera próbki do wizualizacji, zwracając je w dwóch formatach:
        1. Tensory na urządzeniu (do generowania map).
        2. Tablice numpy (do rysowania).
        """
        self.model.eval()
        normalized_tensors_list, denormalized_images_list = [], []
        labels_list, preds_list, image_ids_list = [], [], []

        data_iter = iter(dataloader)
        
        with torch.no_grad():
            for _ in range(int(np.ceil(num_samples / dataloader.batch_size))):
                try:
                    batch_images, batch_labels, batch_ids = next(data_iter)
                    
                    batch_images_device = batch_images.to(self.device)
                    outputs = self.model(batch_images_device)
                    predictions = outputs.argmax(dim=1)
                    
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                    denorm_batch = batch_images * std + mean
                    
                    normalized_tensors_list.append(batch_images_device)
                    denormalized_images_list.append(denorm_batch)
                    labels_list.append(batch_labels.to(self.device))
                    preds_list.append(predictions)
                    image_ids_list.extend(batch_ids)
                except StopIteration:
                    break

        if not normalized_tensors_list:
            empty_tensors = {'images': torch.empty(0), 'labels': torch.empty(0)}
            empty_numpy = {'images': np.array([]), 'labels': np.array([]), 'predictions': np.array([]), 'image_ids': []}
            return empty_tensors, empty_numpy

        # Przygotuj słowniki wyjściowe
        tensors_for_maps = {
            'images': torch.cat(normalized_tensors_list, dim=0)[:num_samples],
            'labels': torch.cat(labels_list, dim=0)[:num_samples]
        }
        numpy_for_plots = {
            'images': torch.cat(denormalized_images_list, dim=0)[:num_samples].cpu().numpy(),
            'labels': torch.cat(labels_list, dim=0)[:num_samples].cpu().numpy(),
            'predictions': torch.cat(preds_list, dim=0)[:num_samples].cpu().numpy(),
            'image_ids': image_ids_list[:num_samples]
        }
        
        return tensors_for_maps, numpy_for_plots

    def _generate_interpretability_maps(self, images_tensor: torch.Tensor, labels_tensor: torch.Tensor) -> List[np.ndarray]:
        """Generuje mapy Grad-CAM lub atencji dla próbek."""
        maps = []
        if self.model_type == 'cnn':
            target_layer = get_cnn_target_layer(self.model)
            if target_layer is None:
                return maps
            
            # Tworzymy instancję GradCAM wewnątrz, aby uniknąć problemów z hookami
            grad_cam = GradCAM(self.model, target_layer)
            for i in range(len(images_tensor)):
                # GradCAM oczekuje pojedynczego obrazu
                heatmap = grad_cam(images_tensor[i].unsqueeze(0), class_idx=labels_tensor[i].item())
                if heatmap is not None:
                    maps.append(heatmap)
            grad_cam.remove_hooks() # Jawne usunięcie hooków
        
        elif self.model_type == 'vit':
            for i in range(len(images_tensor)):
                attention_map = get_vit_attention_map(self.model, images_tensor[i])
                if attention_map is not None:
                    maps.append(attention_map)
        return maps 