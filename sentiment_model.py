from transformers import pipeline
import re

# Cargar el modelo preentrenado multilingüe
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z0-9\s]", "", texto)
    return texto

def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    resultado = classifier(texto_limpio)[0]
    estrellas = int(resultado['label'][0])  # Extrae el número de estrellas

    if estrellas >= 4:
        return "Positiva"
    elif estrellas <= 2:
        return "Negativa"
    else:
        return "Neutral"
