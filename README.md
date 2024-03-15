# Predicción de Fenómenos con Random Forest

Este proyecto utiliza un modelo de Random Forest para predecir fenómenos con base en datos proporcionados. El objetivo es proporcionar una herramienta útil para identificar posibles fenómenos basados en características específicas. El archivo `grafic_3d`.py tiene como objetivo proporcionar una herramienta para la visualización en 3D de datos utilizando el análisis de componentes principales (PCA) y modelos de aprendizaje automático. La visualización en 3D ayuda a comprender mejor la distribución de los datos en un espacio tridimensional, lo que puede ser útil para la exploración y el análisis de conjuntos de datos complejos.

## Contenido del Repositorio

- `modelo_rf.pkl`: Archivo que contiene el modelo entrenado de Random Forest.
- `tfidf_vectorizer.pkl`: Archivo que contiene el vectorizador TF-IDF ajustado.
- El archivo excel debe llamarse `ORIGINAL.xlsx` utilizado para entrenar y evaluar el modelo y contener el campo `RELATO`.
- `grafic_3d`.py: Script de Python que contiene la grafica en 3d utilizando matplotlibt.
- `my_script.py`: Script de Python que contiene la aplicación para realizar predicciones con el modelo entrenado utilizando Streamlit.
- `requirements.txt`: Archivo que lista todas las dependencias del proyecto.

## Instalación

1. Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/ojaviva/classify.git
```

2.  ejecuta la siguiente linea:

```bash
  pip install -r requirements.txt
