import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import multiprocessing as mp
from datetime import datetime

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
DATASET_DIR = CURRENT_PATH + "dataset"   # carpeta raíz con subcarpetas por clase
OUTPUT_MODEL = CURRENT_PATH + "modelo_rust_svm.pkl"
LOG_FILE = CURRENT_PATH + "classification_report.log"
FRAME_SAMPLE_RATE = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CONFIGURACIÓN DE LA MÁSCARA ===
MASK_REGION = {
    "x_ratio": 0.75,   # empieza al 75 % del ancho (parte derecha)
    "y_ratio": 0.75,   # empieza al 75 % de la altura (parte inferior)
    "width_ratio": 0.25,
    "height_ratio": 0.25
}

def mask_face_region(frame):
    """Enmascara la región donde aparece la cámara del jugador."""
    h, w, _ = frame.shape
    x1 = int(w * MASK_REGION["x_ratio"])
    y1 = int(h * MASK_REGION["y_ratio"])
    x2 = int(x1 + w * MASK_REGION["width_ratio"])
    y2 = int(y1 + h * MASK_REGION["height_ratio"])
    frame[y1:y2, x1:x2] = 0
    return frame

# === CARGAR CLIP ===
print(f"🔹 Cargando modelo CLIP ({DEVICE})...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === FUNCIÓN: extraer frames ===
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
            frame = mask_face_region(frame)
            frames.append(frame)
    cap.release()
    return frames

# === FUNCIÓN: obtener embedding ===
def get_video_embedding(video_path):
    frames = extract_frames(video_path)
    if not frames:
        raise ValueError(f"No se pudieron leer frames de {video_path}")
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    return img_feats.mean(dim=0).cpu().numpy()

# === FUNCIÓN paralela ===
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

    NUM_PROCESSES = min(4, mp.cpu_count())
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks)))

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("❌ No se generaron embeddings. Revisa el dataset.")

    X, y = zip(*results)
    X = np.stack(X)
    y = np.array(y)

    # Filtrar clases con menos de 2 muestras
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = [cls for cls, count in zip(unique, counts) if count >= 2]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    classes = [c for i, c in enumerate(classes) if i in valid_classes]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n🔹 Entrenando clasificador (SVM con kernel RBF)...")
    param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.001], "kernel": ["rbf"]}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_

    # === SOLUCIÓN INDEX ERROR ===
    label_to_class = {label: cls for label, cls in zip(np.unique(y_train), classes)}
    class_labels = [label_to_class[i] for i in clf.classes_]

    print(f"\n✅ Mejor configuración: {grid.best_params_}")

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        labels=clf.classes_,
        target_names=class_labels,
        zero_division=0
    )

    print("\n📊 Resultados (TEST):")
    print(report)

    # Guardar el log con fecha
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n=== Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")

    joblib.dump({"clf": clf, "classes": class_labels}, OUTPUT_MODEL)
    print(f"\n✅ Modelo guardado en: {OUTPUT_MODEL}")
    print(f"📝 Reporte añadido a: {LOG_FILE}")
