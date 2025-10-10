
import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

VIDEO_PATH = askopenfilename(title="Selecciona un vídeo para predecir la acción", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.flv *.wmv")])




MASK_REGION = {
    "x_ratio": 0.01,
    "y_ratio": 0.48,
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




cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, 5) # Cogemos el frame número 5 del vídeo
ret, frame = cap.read()
if ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = mask_face_region(frame)
    cv2.imshow("Frame 5", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # mostrar frame
    frames.append(frame)
cap.release()