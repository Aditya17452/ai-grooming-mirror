import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import os
# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Load Model
# -----------------------------
model = torchvision.models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
model_path = os.path.normpath(model_path)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

class_names = ["Formal", "Informal"]

# -----------------------------
# 3. Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# 4. Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Transform
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    label = class_names[predicted.item()]
    conf_score = confidence.item() * 100

    # Display text
    text = f"{label} ({conf_score:.2f}%)"

    cv2.putText(frame, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Formal vs Informal Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()