# UFC Fight Predictor

Proyecto personal de Machine Learning para recopilar y procesar datos históricos
de UFC y, más adelante, entrenar modelos de predicción de combates.

## Fuentes iniciales

- UFCStats
- UFC Record Book
- Ultimate UFC Dataset de Kaggle

## Estado actual

Paso 1: estructura inicial del proyecto.

Todavía no se ha implementado el scraping ni el entrenamiento de modelos.

## Estructura

- `data/raw/`: datos originales, separados por fuente.
- `data/interim/`: datos parcialmente limpiados o combinados.
- `data/processed/`: dataset final preparado para Machine Learning.
- `notebooks/`: exploración y pruebas.
- `src/scraping/`: descarga y extracción de datos.
- `src/processing/`: limpieza y normalización.
- `src/features/`: creación de variables predictivas.
- `src/models/`: entrenamiento y evaluación.
- `tests/`: pruebas automáticas.
