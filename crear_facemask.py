
import cv2
import os
import numpy as np
from tkinter.filedialog import askopenfilename
CURRENT_PATH = os.path.dirname(__file__).replace("\\", "/") + "/"
# VIDEO_PATH = askopenfilename(title="Selecciona un vídeo para crear la facemask", filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.flv *.wmv")])

VIDEO_PATH=CURRENT_PATH+"videos_entrada/2025-10-11 19-13-08.mp4"


# Ajustar para establecer la máscara
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
    { # brújula
        "x_ratio": 0.31,
        "y_ratio": 0,
        "width_ratio": 0.39,
        "height_ratio": 0.04
    },
    { # ping, fps
        "x_ratio": 0,
        "y_ratio": 0.95,
        "width_ratio": 0.1,
        "height_ratio": 0.05
    },
]

def mask_face_region(frame):
    h, w, _ = frame.shape
    for maskRegion in MASK_REGIONS:
        x1 = int(w * maskRegion["x_ratio"])
        y1 = int(h * maskRegion["y_ratio"])
        x2 = int(x1 + w * maskRegion["width_ratio"])
        y2 = int(y1 + h * maskRegion["height_ratio"])
        frame[y1:y2, x1:x2] = 0
    return frame




cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, 50000) # Cogemos el frame número 5 del vídeo
ret, frame = cap.read()
if ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = mask_face_region(frame)
    cv2.imshow("Frame 5", frame)
    # cv2.imwrite(filename=CURRENT_PATH+"frameout.png", img=frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # mostrar frame
    frames.append(frame)
cap.release()