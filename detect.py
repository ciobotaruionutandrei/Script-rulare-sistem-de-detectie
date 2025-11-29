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

# Funcție IoU pentru detectarea suprapunerilor
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Lista cu detectiile finale
final_boxes = []

# Mai întâi detectăm semafor galben (prioritar)
for model_path in models:
    if "semfor galben" in model_path.lower():  # detectam doar semafor galben
        model = YOLO(model_path)
        results = model(image_path)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            final_boxes.append({'box':[x1,y1,x2,y2], 'label':label, 'conf':conf})

# Apoi detectăm restul modelelor, dar doar dacă nu se suprapun cu semafor galben
for model_path in models:
    if "semfor galben" in model_path.lower():
        continue  # deja procesat
    model = YOLO(model_path)
    results = model(image_path)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Verificăm suprapunerea cu semafor galben
        overlap = False
        for b in final_boxes:
            if b['label'] == "Semafor galben" and iou([x1,y1,x2,y2], b['box']) > 0.5:
                overlap = True
                break
        if not overlap:
            final_boxes.append({'box':[x1,y1,x2,y2], 'label':label, 'conf':conf})

# Desenăm toate detectările finale pe imagine
for b in final_boxes:
    x1, y1, x2, y2 = b['box']
    label = b['label']
    conf = b['conf']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Salvăm imaginea finală
output_image_path = os.path.join(output_folder, "rezultat_combinat.jpg")
cv2.imwrite(output_image_path, img)

print(f"Rezultatele finale au fost salvate într-o singură imagine: {output_image_path}")
