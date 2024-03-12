import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from random import randint
import joblib
import streamlit as st
import nltk
from nltk.corpus import stopwords

# Asegúrate de tener las stopwords en español disponibles
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Carga de los datos
df_original = pd.read_excel("ORIGINAL.xlsx")

# Concatenar los campos delito, departamento, municipio y relato en un nuevo campo llamado 'relato'
df_original['RELATO'] = df_original['DELITO'] + ' ' + df_original['DEPARTAMENTO_HECHO'] + ' ' + df_original['MUNICIPIO_HECHO'] + ' ' + df_original['RELATO']

# Eliminar los campos delito, departamento y municipio
df_original.drop(['DELITO', 'DEPARTAMENTO_HECHO', 'MUNICIPIO_HECHO'], axis=1, inplace=True)

# División de los datos en conjuntos de entrenamiento, prueba y validación
df_train_test, df_valid = train_test_split(df_original, test_size=0.2, random_state=42, stratify=df_original['Fenomeno'])
df_train, df_test = train_test_split(df_train_test, test_size=0.25, random_state=42, stratify=df_train_test['Fenomeno'])
# Imprimir los subconjuntos de datos divididos
print("\nConjunto de datos de entrenamiento:", len(df_train))
print("Conjunto de datos de prueba:", len(df_test))
print("Conjunto de datos de validación:", len(df_valid))

print("\n----- Archivo Cargado--------")

# Inicializa el vectorizador TF-IDF con las stopwords en español
tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=980)

# Ajusta el vectorizador a los datos de entrenamiento y transforma los datos de entrenamiento
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['RELATO'].astype('U'))

# Guardado del vectorizador TF-IDF ajustado
model_filename_1 = 'tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, model_filename_1)

# Manejo de Desbalance de Clases con SMOTE
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_smote, y_train_balanced = smote.fit_resample(X_train_tfidf, df_train['Fenomeno'])
print("----- Balanceo de los Datos--------")

# Modelado con Random Forest
modelo_rf = RandomForestClassifier(n_estimators=360,  # Incrementa el número de árboles en el bosque
                                   max_depth=randint(35,42),      # Aumenta la profundidad máxima de cada árbol 
                                   min_samples_split=3,  # Incrementa el número mínimo de muestras requeridas para dividir un nodo interno
                                   min_samples_leaf=2,   # Incrementa el número mínimo de muestras requeridas para estar en un nodo hoja
                                   random_state=42)
#(n_estimators=100, max_depth=None, random_state=42)
modelo_rf.fit(X_train_smote, y_train_balanced)

# Predicciones y evaluación del modelo
X_test_tfidf = tfidf_vectorizer.transform(df_test['RELATO'].values.astype('U'))
y_pred_test = modelo_rf.predict(X_test_tfidf)

# Evaluación del modelo
precision = accuracy_score(df_test['Fenomeno'], y_pred_test)
print("Precisión del modelo Random Forest:", precision)
print("\nInforme de clasificación:")
print(classification_report(df_test['Fenomeno'], y_pred_test))
print("\nMatriz de confusión:")
print(confusion_matrix(df_test['Fenomeno'], y_pred_test))
print("----- Presicion de los datos--------")

# Guardar el modelo entrenado en un archivo .pkl
model_filename = 'modelo_rf.pkl'
joblib.dump(modelo_rf, model_filename)
model_filename

# Asumiendo que 'modelo_rf' y 'X_valid_tfidf' están definidos correctamente

# Codificar las etiquetas de validación
label_encoder = LabelEncoder()
y_valid_encoded = label_encoder.fit_transform(df_valid['Fenomeno'])

# Realizar predicciones con el modelo sobre los datos de validación transformados
X_valid_tfidf = tfidf_vectorizer.transform(df_valid['RELATO'].values.astype('U'))
y_pred_valid = modelo_rf.predict(X_valid_tfidf)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(df_valid['Fenomeno'], y_pred_valid)
print("Precisión en los datos de validación:", accuracy)
print("Informe de clasificación:\n", classification_report(df_valid['Fenomeno'], y_pred_valid))
print("Matriz de confusión:\n", confusion_matrix(df_valid['Fenomeno'], y_pred_valid))
print("----- Muestra el resultado-------")

# Cargar el modelo y el vectorizador desde archivos
modelo_rf = joblib.load('modelo_rf.pkl')

def predict(texts):
    # Transforma los textos a su representación TF-IDF
    texts_tfidf = tfidf_vectorizer.transform(texts)
    # Realiza predicciones
    predictions = modelo_rf.predict(texts_tfidf)
    return predictions

# Creando la interfaz de usuario con Streamlit
st.title('Predicción con Modelo Random Forest')

# Subida de archivo
uploaded_file = st.file_uploader("Elige un archivo Excel con los registros a predecir", type=['xlsx'])

if uploaded_file is not None:
    # Cargar el archivo Excel
    df_to_predict = pd.read_excel(uploaded_file)
    # Asumiendo que la columna con texto a predecir se llama 'RELATO'
    if 'RELATO' in df_to_predict.columns:
        # Concatenar campos delito, departamento, municipio y relato en 'relato'
        df_to_predict['RELATO'] = df_to_predict['DELITO'] + ' ' + df_to_predict['DEPARTAMENTO_HECHO'] + ' ' + df_to_predict['MUNICIPIO_HECHO'] + ' ' + df_to_predict['RELATO']
        texts = df_to_predict['RELATO'].astype('U').tolist()
        predictions = predict(texts)
        df_to_predict['Predicciones'] = predictions
        st.write(df_to_predict)
    else:
        st.error("El archivo debe tener una columna 'RELATO' con los textos a predecir.")
print("----- se empaqueta para usar el Modelo Random Forest1--------")
