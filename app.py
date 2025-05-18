from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_model import predecir_sentimiento

# --- FUNCIONES DEL SISTEMA CBR ---

def extraer_asin(url):
    match = re.search(r'/dp/([A-Z0-9]{10})', url)
    if match:
        return match.group(1)
    return None

def cargar_productos(filepath, max_items=50000):
    productos = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            try:
                data = json.loads(line)
                if 'asin' in data and 'title' in data and data['title']:
                    productos.append({
                        'asin': str(data.get('asin', '')),
                        'title': str(data.get('title', '')),
                        'brand': str(data.get('brand', '')),
                        'category': ' '.join(data.get('category', [])) if isinstance(data.get('category'), list) else str(data.get('category', ''))
                    })
            except:
                continue
    return pd.DataFrame(productos)

def recomendar_similares_por_texto(titulo, marca, categoria, df, top_n=5):
    df = df.fillna('').copy()
    df['title'] = df['title'].astype(str)
    df['brand'] = df['brand'].astype(str)
    df['category'] = df['category'].astype(str)

    consulta = f"{titulo} {marca} {categoria}"
    corpus = [consulta] + list(df['title'] + " " + df['brand'] + " " + df['category'])

    # Mejor vectorizador con limpieza
    tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2).fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]

    similares = df.iloc[top_indices].copy()
    similares['sim_score'] = sims[top_indices]
    return similares[['asin', 'title', 'brand', 'sim_score']]

def extraer_info_producto(driver):
    title, brand, category = "", "", ""

    try:
        title = driver.find_element(By.ID, "productTitle").text.strip()
    except:
        pass

    try:
        brand = driver.find_element(By.ID, "bylineInfo").text.strip()
    except:
        try:
            rows = driver.find_elements(By.CSS_SELECTOR, "#productDetails_techSpec_section_1 tr")
            for row in rows:
                if "Marca" in row.text or "Brand" in row.text:
                    brand = row.find_elements(By.TAG_NAME, "td")[0].text.strip()
                    break
        except:
            pass

    try:
        category_elements = driver.find_elements(By.CSS_SELECTOR, "ul.a-unordered-list.a-horizontal.a-size-small li")
        category = ' '.join([el.text.strip() for el in category_elements if el.text.strip()])
    except:
        pass

    return title, brand, category

# --- FLASK APP CONFIG ---

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
CBR_DATASET_PATH = os.path.join(BASE_DIR, "dataset_limpio.jsonl")
cb_df = cargar_productos(CBR_DATASET_PATH, max_items=50000)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')

    reviews = scrape_amazon_reviews(url)
    print("🔎 Reseñas extraídas:", reviews)

    if not reviews:
        return jsonify({
            "reviews": [],
            "positive_percentage": 0,
            "negative_percentage": 0
        })

    results = []
    positivas = 0
    negativas = 0

    for review in reviews:
        sentimiento = predecir_sentimiento(review)
        print(f"📝 Reseña: {review}\n➡️ Sentimiento: {sentimiento}")

        if sentimiento.lower().startswith("pos"):
            sentimiento_clean = "positive"
            positivas += 1
        else:
            sentimiento_clean = "negative"
            negativas += 1

        results.append({"text": review, "sentiment": sentimiento_clean})

    total = positivas + negativas
    porcentaje_positivas = round((positivas / total) * 100, 2) if total > 0 else 0
    porcentaje_negativas = round((negativas / total) * 100, 2) if total > 0 else 0

    response_json = {
        "reviews": results,
        "positive_percentage": porcentaje_positivas,
        "negative_percentage": porcentaje_negativas
    }

    print("📦 JSON devuelto:", response_json)
    return jsonify(response_json)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    data = request.get_json()
    url = data.get('url')

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)

    title, brand, category = extraer_info_producto(driver)
    driver.quit()

    if not title:
        return jsonify({"error": "No se pudo obtener información del producto."}), 400

    print("📌 Producto base extraído:", title, brand, category)

    similares = recomendar_similares_por_texto(title, brand, category, cb_df, top_n=5)
    resultados = similares.to_dict(orient="records")

    for r in resultados:
        r["url"] = f"https://www.amazon.com/dp/{r['asin']}" if r['asin'] else "#"

    return jsonify(resultados)

def scrape_amazon_reviews(url, max_reviews=5):
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)

    try:
        reviews_link = driver.find_element(By.PARTIAL_LINK_TEXT, "See all reviews")
        reviews_link.click()
        time.sleep(2)
    except:
        pass

    review_elements = driver.find_elements(By.CLASS_NAME, 'review-text-content')
    reviews = [el.text.strip() for el in review_elements[:max_reviews]]
    driver.quit()
    return reviews

if __name__ == '__main__':
    app.run(debug=True)
