import os
import json
import re
print("Cargando librerías...")
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import joblib

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "acciones.json"
PRINT_RESULTS = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CARGAR MODELO ===
print(f"🔹 Cargando modelo CLIP ({DEVICE}) y clasificador SVM...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
data = joblib.load(MODEL_PATH)
clf, classes = data["clf"], data["classes"]

# === CARGAR ACCIONES ===
with open(ACTIONS_JSON, "r", encoding="utf-8") as f:
    ACTIONS_MAP = json.load(f)

# === FUNCIÓN: detectar duración del prompt ===
def parse_duration_from_prompt(prompt):
    prompt = prompt.lower()
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(min|minute|minuto|seg|segundo|s)", prompt)
    if not match:
        return 60
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    return int(value * 60) if unit.startswith("min") else int(value)

# === FUNCIÓN: determinar clase más parecida al prompt ===
def find_most_similar_class(prompt, min_similarity=0.25):
    text_list = []
    class_lookup = []
    for cls, phrases in ACTIONS_MAP.items():
        for phrase in phrases:
            text_list.append(phrase)
            class_lookup.append(cls)
    inputs = processor(text=text_list, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    prompt_inputs = processor(text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        prompt_feat = model.get_text_features(**prompt_inputs)
    prompt_feat = prompt_feat / prompt_feat.norm(p=2, dim=-1, keepdim=True)
    similarities = (prompt_feat @ text_feats.T).cpu().numpy()[0]
    best_idx = np.argmax(similarities)
    best_class = class_lookup[best_idx]
    best_phrase = text_list[best_idx]
    best_score = float(similarities[best_idx])
    if best_score < min_similarity:
        print(f"⚠️ Ninguna clase es suficientemente similar (máx={best_score:.3f}).")
        return None, best_score, best_phrase
    print(f"\n🧠 Frase más similar: “{best_phrase}” ({best_score:.3f}) → Clase: {best_class}")
    if PRINT_RESULTS:
        lastClass = None
        for index, pctSimilar in enumerate(similarities):
            clase = class_lookup[index]
            texto = text_list[index]
            if clase == lastClass:
                clase = ""
            else:
                lastClass = clase
            if texto == best_phrase:
                selected = "*"
            else:
                selected = ""
            print(f"{clase:20.20} - {texto:50.50} - {pctSimilar:.5f} {selected}")
            # results[pctSimilar] = {"clase": class_lookup[index], "texto": text_list[index]}
    
    return best_class, best_score, best_phrase


# === MAIN ===
if __name__ == "__main__":
    while True:
        prompt = input("💬 Prompt (Ctrl+C para salir): ").strip()
        clip_duration = parse_duration_from_prompt(prompt)
        print(f"⏱️ Duración detectada: {clip_duration} segundos")
        target_class, score, phrase = find_most_similar_class(prompt)
        if not target_class:
            print("No se ha detectado ninguna clase")
        print(f"🎯 Clase seleccionada: {target_class} (similitud {score:.3f})")
