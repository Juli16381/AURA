# limpiar_dataset.py
import json
import os

entrada = "C:/Users/gabri/Desktop/AURA/AURA/meta_Clothing_Shoes_and_Jewelry.jsonl"
salida = os.path.join("AURA", "dataset_limpio.jsonl")  # guarda dentro de carpeta raíz AURA
max_items = 50000

productos_validos = 0

with open(entrada, 'r', encoding='utf-8') as f_in, open(salida, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if productos_validos >= max_items:
            break
        try:
            data = json.loads(line)

            # Extraer campos con tolerancia a faltantes
            asin = data.get("parent_asin") or data.get("asin")
            title = data.get("title", "").strip()
            brand = (data.get("store") or "").strip()

            categories = data.get("categories", []) or data.get("category", [])

            if asin and title:  # Solo exigimos ASIN y título
                producto = {
                    "asin": asin,
                    "title": title,
                    "brand": brand,
                    "category": categories
                }
                f_out.write(json.dumps(producto) + "\n")
                productos_validos += 1

                if productos_validos % 1000 == 0:
                    print(f"✅ Procesados {productos_validos} productos...")

        except json.JSONDecodeError:
            continue

print(f"🎯 Finalizado. Total guardados: {productos_validos} en {salida}")
