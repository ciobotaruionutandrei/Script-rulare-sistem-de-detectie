from ultralytics import YOLO
import cv2
import os

# --- Setări --- #
image_path = r"C:\Users\usera228\Desktop\Test licenta\Captura de ecran 2025-11-28 212433.jpg"
output_folder = r"C:\Users\usera228\Desktop\Test licenta\rezultate"
os.makedirs(output_folder, exist_ok=True)

# Imaginea inițială
img = cv2.imread(image_path)

# Lista modelelor tale
models = [
    r"C:\Users\usera228\Desktop\Test licenta\Model accesul interzis.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model cedează trecerea nano.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model drum cu prioritate.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Limitare de viteză.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model lucrari.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Oprire interzisă.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model semafor roșu.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Semafor verde.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model semfor galben.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Sens Giratoriu.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Sens unic.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model Stop.pt",
    r"C:\Users\usera228\Desktop\Test licenta\Model trecere de pietoni.pt"
]

# Pentru fiecare model, facem detectarea și desenăm rezultatele pe aceeași imagine
for model_path in models:
    model = YOLO(model_path)
    results = model(image_path)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Desenăm box-ul și label-ul
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Salvează imaginea finală cu toate detectările
output_image_path = os.path.join(output_folder, "rezultat_combinat.jpg")
cv2.imwrite(output_image_path, img)

print(f"Toate detectările au fost salvate într-o singură imagine: {output_image_path}")
