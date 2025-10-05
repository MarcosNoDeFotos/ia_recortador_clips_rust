import torch, cv2, joblib, numpy as np
from transformers import CLIPProcessor, CLIPModel
import os



CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", weights_only=False).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Cargar clasificador
data = joblib.load(CURRENT_PATH+"modelo_rust_svm.pkl")
clf = data["clf"]
classes = data["classes"]

def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def predict_action(video_path):
    frames = extract_frames(video_path)
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    embedding = img_feats.mean(dim=0).cpu().numpy()
    pred = clf.predict([embedding])[0]
    return classes[pred]

# Ejemplo
print("Predicción:", predict_action(CURRENT_PATH+"nuevo_clip.mkv"))
