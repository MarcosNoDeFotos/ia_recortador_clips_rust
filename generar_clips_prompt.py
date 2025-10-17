import os
import json
import re
import cv2
print("Cargando librerías...")
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import normalize
import joblib
import subprocess
from multiprocessing import Pool, cpu_count

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "modelos/modelo_rust_svm.pkl"
ACTIONS_JSON = CURRENT_PATH + "acciones.json"
VIDEOS_INPUT_DIR = CURRENT_PATH + "videos_entrada/"
OUTPUT_DIR = CURRENT_PATH + "clips_generados/"
FRAMES_CACHE_DIR = os.path.join(OUTPUT_DIR, "frames_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEOS_INPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_CACHE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === REGIONES A ENMASCARAR ===
MASK_REGIONS = [
    { # Cámara
        "x_ratio": 0.01,
        "y_ratio": 0.48,
        "width_ratio": 0.25,
        "height_ratio": 0.25
    },
    { # Salud, comida, agua
        "x_ratio": 0.844,
        "y_ratio": 0.87,
        "width_ratio": 0.5,
        "height_ratio": 0.5
    },
    { # Brújula
        "x_ratio": 0.31,
        "y_ratio": 0.0,
        "width_ratio": 0.39,
        "height_ratio": 0.04
    },
    { # Ping, FPS
        "x_ratio": 0.0,
        "y_ratio": 0.95,
        "width_ratio": 0.1,
        "height_ratio": 0.05
    },
]

def mask_face_region(frame):
    """Aplica máscaras sobre HUD, cámara y otros elementos estáticos."""
    h, w, _ = frame.shape
    for mask in MASK_REGIONS:
        x1 = int(w * mask["x_ratio"])
        y1 = int(h * mask["y_ratio"])
        x2 = int(x1 + w * mask["width_ratio"])
        y2 = int(y1 + h * mask["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame

# === CARGAR MODELO ===
print(f"🔹 Cargando modelo CLIP ({DEVICE}) y clasificador SVM...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
data = joblib.load(MODEL_PATH)
clf, classes = data["clf"], data["classes"]

# === CARGAR ACCIONES ===
with open(ACTIONS_JSON, "r", encoding="utf-8") as f:
    ACTIONS_MAP = json.load(f)

# === FUNCIÓN: extraer frames con caché ===
def extract_frames(video_path, fps=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(FRAMES_CACHE_DIR, video_name)
    os.makedirs(cache_dir, exist_ok=True)

    # Leer caché si existe
    cached_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".npy")])
    if cached_files:
        frames, timestamps = [], []
        for f in tqdm(cached_files, desc=f"Cargando frames cache de {video_name}", unit="frame"):
            data = np.load(os.path.join(cache_dir, f), allow_pickle=True)
            frames.append(data[0])
            timestamps.append(data[1])
        return frames, timestamps

    # Extraer frames si no hay caché
    cap = cv2.VideoCapture(video_path)
    frames, timestamps = [], []
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(video_fps // fps))

    for i in tqdm(range(0, total_frames, step), desc=f"Extrayendo frames de {video_name}", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = mask_face_region(frame_rgb)  # 🔹 Aplicar máscara aquí
        frames.append(frame_rgb)
        timestamp = i / video_fps
        timestamps.append(timestamp)

        # Guardar frame en caché
        cache_file = os.path.join(cache_dir, f"{i:08d}.npy")
        np.save(cache_file, np.array([frame_rgb, timestamp], dtype=object))

    cap.release()
    return frames, timestamps

# === FUNCIÓN: embedding de un frame ===
def get_frame_embedding(frame):
    inputs = processor(images=[frame], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = normalize(feats.cpu().numpy())
    return feats[0]

# === FUNCIÓN: embedding en paralelo ===
def get_embeddings_parallel(frames):
    print("\n🔹 Extrayendo embeddings de frames...")
    with Pool(processes=min(cpu_count(), 4)) as pool:
        embeddings = list(tqdm(pool.imap(get_frame_embedding, frames), total=len(frames), unit="frame"))
    return np.array(embeddings)

# === FUNCIÓN: detectar duración del prompt ===
def parse_duration_from_prompt(prompt):
    prompt = prompt.lower()
    match = re.search(r"(\d+(?:[\.,]\d+)?)\s*(min|minute|minuto|seg|segundo|s)", prompt)
    if not match:
        return 60
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2)
    return int(value * 60) if unit.startswith("min") else int(value)

# === FUNCIÓN: encontrar clase más similar ===
def find_most_similar_class(prompt, min_similarity=0.25):
    text_list, class_lookup = [], []
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
    return best_class, best_score, best_phrase

# === FUNCIÓN: generar clips ===
def generar_clips(video_path, prompt, threshold=0.25):
    print(f"\n🎯 Procesando vídeo: {os.path.basename(video_path)}")
    print(f"💬 Prompt: {prompt}")
    clip_duration = parse_duration_from_prompt(prompt)
    print(f"⏱️ Duración detectada: {clip_duration} segundos")

    target_class, score, phrase = find_most_similar_class(prompt)
    if not target_class:
        return
    print(f"🎯 Clase seleccionada: {target_class} (similitud {score:.3f})")
    input("Pulsa intro para generar clips con esta clase")

    # Extraer frames con caché
    frames, timestamps = extract_frames(video_path, fps=2)

    # Embeddings en paralelo
    embeddings = get_embeddings_parallel(frames)

    # Predicción de probabilidad de la clase
    probs = clf.predict_proba(embeddings)
    class_idx = classes.index(target_class)
    target_probs = probs[:, class_idx]

    # Encontrar segmentos por umbral
    segments, start, segment_probs = [], None, []
    for i, prob in enumerate(target_probs):
        if prob > threshold and start is None:
            start = timestamps[i]
            segment_probs = [prob]
        elif prob > threshold and start is not None:
            segment_probs.append(prob)
        elif prob <= threshold and start is not None:
            end = timestamps[i]
            avg_prob = np.mean(segment_probs)
            if end - start >= clip_duration / 6:
                segments.append((start, end, avg_prob))
            start = None
            segment_probs = []

    if start is not None and segment_probs:
        end = timestamps[-1]
        avg_prob = np.mean(segment_probs)
        if end - start >= clip_duration / 6:
            segments.append((start, end, avg_prob))

    if not segments:
        print("⚠️ No se encontraron segmentos que coincidan con el prompt.")
        return

    # Generar clips con ffmpeg
    for idx, (start, end, avg_prob) in enumerate(segments):
        middle = (start + end) / 2
        clip_start = max(0, middle - clip_duration / 2)
        avg_prob_rounded = round(avg_prob, 3)
        output_path = os.path.join(
            OUTPUT_DIR,
            f"{os.path.splitext(os.path.basename(video_path))[0]}_{target_class}_{idx+1}_{avg_prob_rounded}.mp4"
        )
        cmd = [
            "ffmpeg", "-y", "-ss", str(clip_start), "-i", video_path,
            "-t", str(clip_duration), "-map", "0", "-c", "copy", output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎬 Clip generado: {output_path}  |  treshold detectado: {avg_prob:.3f}")

# === MAIN ===
if __name__ == "__main__":
    prompt = input("💬 Prompt: ").strip()
    video_files = [f for f in os.listdir(VIDEOS_INPUT_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))]
    if not video_files:
        print("❌ No se encontraron vídeos en la carpeta videos_entrada/")
    for vf in video_files:
        video_path = os.path.join(VIDEOS_INPUT_DIR, vf)
        generar_clips(video_path, prompt)
