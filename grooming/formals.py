import torch
import timm
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
import os
import sys

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_formal_classifier.pth")
   # your .pth file
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# LOAD MODEL ARCHITECTURE
# ==========================================================
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

# Load weights
if not os.path.exists(MODEL_PATH):
    print("Model file not found.")
    sys.exit()

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

# ==========================================================
# IMAGE TRANSFORM
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ==========================================================
# CAMERA INIT
# ==========================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    sys.exit()

print("Press Q to exit")

prev_time = 0

# ==========================================================
# MAIN LOOP
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame to PIL
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    label = "FORMAL - Allowed" if predicted.item() == 0 else "NOT FORMAL - Rejected"
    color = (0,255,0) if predicted.item() == 0 else (0,0,255)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.putText(frame, f"FPS: {int(fps)}", (30,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Dress Code Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
