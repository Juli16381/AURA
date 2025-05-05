from flask import Flask, render_template, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from sentiment_model import predecir_sentimiento

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    url = data.get('url')
    
    # Extraer reseñas
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
        
        # Homogenizamos los nombres para frontend (solo "positive" o "negative")
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



#TRADUCIR
#QUE SIRVA LO DE DESCARGAR PDF
