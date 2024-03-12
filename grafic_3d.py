import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar el modelo y el vectorizador
modelo_rf = joblib.load('modelo_rf.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Generar datos aleatorios (solo para propósitos de ejemplo)
X_train_smote_fake = np.random.rand(100, 1000)

# Transformar datos con PCA
X_train_pca = PCA(n_components=3).fit_transform(X_train_smote_fake)

# Realizar predicciones sobre los datos transformados (solo para propósitos de ejemplo)
predictions_fake = np.random.randint(0, 2, size=(100,))

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Asignar colores según las predicciones (solo para propósitos de ejemplo)
colors = np.where(predictions_fake == 1, 'r', 'b')

# Visualizar datos en 3D
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=colors)

# Configuraciones adicionales de visualización
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('Visualización en 3D con PCA')

plt.show()
