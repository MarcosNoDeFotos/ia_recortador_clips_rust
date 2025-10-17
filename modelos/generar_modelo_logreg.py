import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import multiprocessing as mp

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
DATASET_DIR = CURRENT_PATH + "dataset"   # carpeta raíz con subcarpetas por clase
OUTPUT_MODEL = CURRENT_PATH + "modelo_rust_logreg.pkl"
FRAME_SAMPLE_RATE = 30    # nº de frames que se extraen por vídeo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CARGAR CLIP ===
print(f"🔹 Cargando modelo CLIP ({DEVICE})...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === FUNCIÓN: extraer frames representativos ===
def extract_frames(video_path, num_frames=FRAME_SAMPLE_RATE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

# === FUNCIÓN: obtener embedding promedio de un vídeo ===
def get_video_embedding(video_path):
    frames = extract_frames(video_path)
    if not frames:
        raise ValueError(f"No se pudieron leer frames de {video_path}")
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    return img_feats.mean(dim=0).cpu().numpy()

# === FUNCIÓN para procesamiento en paralelo ===
def process_video(args):
    cls, file, label = args
    if not file.lower().endswith((".mp4", ".mov", ".avi")):
        return None
    path = os.path.join(DATASET_DIR, cls, file)
    try:
        emb = get_video_embedding(path)
        return emb, label
    except Exception as e:
        print(f"⚠️ Error procesando {file}: {e}")
        return None

# === MAIN ===
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    classes = sorted(os.listdir(DATASET_DIR))
    print(f"🔹 Clases detectadas: {classes}\n")

    # Crear lista de tareas
    tasks = []
    for label, cls in enumerate(classes):
        cls_path = os.path.join(DATASET_DIR, cls)
        for file in os.listdir(cls_path):
            tasks.append((cls, file, label))

    print(f"🔹 Procesando {len(tasks)} vídeos en paralelo...")

    NUM_PROCESSES = min(4, mp.cpu_count())  # ajusta según tu CPU/GPU
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks)))

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("❌ No se generaron embeddings. Revisa el dataset.")

    X, y = zip(*results)
    X = np.stack(X)
    y = np.array(y)

    # === División entrenamiento / prueba ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Comprobamos si alguna clase no aparece en test
    test_classes_present = set(y_test)
    missing = [cls for i, cls in enumerate(classes) if i not in test_classes_present]
    if missing:
        print(f"⚠️ Clases sin muestras en TEST: {missing}")

    # === ENTRENAR CLASIFICADOR ===
    print("\n🔹 Entrenando clasificador (Logistic Regression)...")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)

    # === EVALUAR ===
    y_pred = clf.predict(X_test)
    print("\n📊 Resultados (TEST):")
    print(classification_report(
        y_test, y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    ))

    # === GUARDAR MODELO ===
    joblib.dump({"clf": clf, "classes": classes}, OUTPUT_MODEL)
    print(f"\n✅ Modelo guardado en: {OUTPUT_MODEL}")
