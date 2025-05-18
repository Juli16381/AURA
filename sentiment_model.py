import os
import joblib
import re

# Asegura que la ruta sea relativa al archivo actual, no al directorio de ejecución
BASE_DIR = os.path.dirname(__file__)
modelo_path = os.path.join(BASE_DIR, "modelos", "modelo_naive_bayesULTIMO.pkl")
vectorizer_path = os.path.join(BASE_DIR, "modelos", "vectorizer_tfidfULTIMO.pkl")

# Carga del modelo y vectorizador
vectorizer = joblib.load(vectorizer_path)
modelo = joblib.load(modelo_path)

# Si tu modelo usa un vectorizador, también debes cargarlo (por ejemplo: CountVectorizer o TfidfVectorizer)
# Si tu modelo ya lo tiene incluido en un pipeline, no necesitas esto.
# Supongamos que es un pipeline completo:
# Ej: modelo = joblib.load("modelos/modelo_naive_bayesULTIMO.pkl")

def limpiar_texto(texto):
    texto = texto.lower()  # Convertir el texto a minúsculas
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)  # Eliminar caracteres no alfanuméricos
    return texto

def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)  # Limpiar el texto
    vector = vectorizer.transform([texto_limpio])  # Convertir a formato adecuado para el modelo (2D)
    
    # Realizar la predicción
    prediccion = modelo.predict(vector)[0]
    
    # Asegurarse de que la predicción solo sea positiva o negativa
    if prediccion == 1:
        return "Positiva"
    else:
        return "Negativa"
