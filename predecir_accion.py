import os
import cv2
import torch
import numpy as np
import joblib
from transformers import CLIPProcessor, CLIPModel
from tkinter.filedialog import askopenfilename

# === CONFIGURACIÓN ===
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
MODEL_PATH = CURRENT_PATH + "modelos/modelo_rust_svm.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SAMPLE_RATE = 30

OPENED_VIDEO_FILE = askopenfilename(title="Selecciona un vídeo para predecir la acción", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.flv *.wmv")])


# === CONFIGURACIÓN DE LA MÁSCARA ===
MASK_REGION = {
    "x_ratio": 0.75,
    "y_ratio": 0.75,
    "width_ratio": 0.25,
    "height_ratio": 0.25
}

def mask_face_region(frame):
    h, w, _ = frame.shape
    x1 = int(w * MASK_REGION["x_ratio"])
    y1 = int(h * MASK_REGION["y_ratio"])
    x2 = int(x1 + w * MASK_REGION["width_ratio"])
    y2 = int(y1 + h * MASK_REGION["height_ratio"])
    frame[y1:y2, x1:x2] = 0
    return frame

# === CARGAR MODELO Y CLIP ===
print(f"🔹 Cargando modelo CLIP ({DEVICE})...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

data = joblib.load(MODEL_PATH)
clf = data["clf"]
classes = data["classes"]

# === FUNCIÓN: extraer frames ===
def extract_frames(video_path, num_frames=FRAME_SAMPLE_RATE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int)
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
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    return img_feats.mean(dim=0).cpu().numpy()

# === PREDICCIÓN ===
# video_path = input("🎥 Introduce la ruta del vídeo a analizar: ").strip().strip('"')
print(f"\n🔹 Analizando: {OPENED_VIDEO_FILE}")

embedding = get_video_embedding(OPENED_VIDEO_FILE)
proba = clf.predict_proba([embedding])[0]
pred_idx = np.argmax(proba)
pred_label = classes[pred_idx]

print(f"\n✅ Acción detectada: {pred_label}")
print("📊 Probabilidades:")
for cls, p in zip(classes, proba):
    print(f" - {cls:40}: {p:.4f}")
