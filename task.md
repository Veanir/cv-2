# Projekt zaliczeniowy - Sieci neuronowe

## Opis - Hipotezy badawcze

### Przykłady hipotez:

- Dodanie warstw Batch Normalization przyspiesza proces uczenia się sieci neuronowej dla klasyfikacji obrazów.
- Wykorzystanie transfer learningu (np. modelu wstępnie wytrenowanego na ImageNet) poprawia dokładność modelu w zadaniu klasyfikacji medycznych zdjęć RTG.
- Zwiększenie rozdzielczości obrazów wejściowych powyżej pewnego progu nie poprawia istotnie dokładności, a znacząco wydłuża czas trenowania.

Wybierz jedną z hipotez badawczych, zmodyfikuj ją lub zaproponuj własną oraz spróbuj ją udowodnić bądź obalić. Zaprojektuj odpowiednie eksperymenty badawcze, przygotuj modele, zbiory danych oraz wszystkie inne potrzebne elementy wymagane do zweryfikowania hipotezy badawczej.

## Wymagania

Przygotowanie raportu w formie posteru zawierającego:

- **Abstract**
- **Introduction**
- **Related Work**
- **Data**
- **Methods**
- **Experiments**
- **Conclusion**
- **Supplementary Material** (opcjonalnie)

### W powyższych sekcjach musi zawierać się:

- przygotowanie oraz prezentacja wybranego zbioru danych (dowolnego, opartego na danych wizualnych)
- przygotowanie oraz prezentacja głównych metod
- prezentacja wyników eksperymentów
- **(!) przygotowanie bibliografii** - wszelkie cytowane prace naukowe i opracowania (!), wliczając w to strony internetowe

## Warianty projektu

### Wariant na maksymalną ocenę 3.0:

- skorzystanie z gotowej sieci neuronowej, dostępnej w Keras API, TensorFlow Hub, PyTorch Models, Huggingface itd.
- Wykonanie podstawowych eksperymentów, próbujących zweryfikować wybraną hipotezę badawczą
- Przygotowanie postera
- Przygotowanie kodu w postaci notebooka/repozytorium

### Przykładowe wymagania na wyższą ocenę:

- zaprojektowanie własnej architektury sieci neuronowej/eksperymentów, wykorzystanie zbiorów danych
- przeprowadzenie procesu uczenia sieci neuronowej
- zaproponowanie ciekawej hipotezy badawczej
- porównanie różnych architektur
- analiza wpływu parametrów
- ablation study
- dokładna analiza wyników (krzywe uczenia, analiza błędów, confusion matrix)
- wykorzystanie literatury
- fine-tuning pretrenowanego modelu
- użycie zaawansowanych modeli (np. Vision Transformer)
- eksperymenty z generalizacją (inne zbiory danych, transfer learning)
- własny zbiór danych lub nietypowy problem (np. segmentacja, super resolution)

# Wybrana hipoteza:
```
Vision Transformers osiągają lepszą dokładność niż tradycyjne CNN w klasyfikacji zdjęć dermatologicznych, szczególnie przy ograniczonych danych treningowych
```

Porównanie nowoczesnych architektur (ViT vs ResNet/EfficientNet)
Zastosowanie w medycynie (wysokie znaczenie praktyczne)
Możliwość badania transfer learning z różnych pre-trained models
Analiza wpływu wielkości datasetu na performance