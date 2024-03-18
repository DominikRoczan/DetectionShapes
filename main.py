from ultralytics import YOLO

#
# model = YOLO('yolov8n.yaml')
#
# results = model.train(data='coco128.yaml', epochs=3)

import cv2
import numpy as np

# Ścieżka do plików konfiguracyjnych modelu YOLO
yolo_config_path = 'yolov3.cfg'
yolo_weights_path = 'yolov3.weights'
yolo_labels_path = 'dataset/labels'  # Plik z etykietami dla modelu YOLO

# Wczytanie etykiet
with open(yolo_labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Inicjalizacja kolorów losowych dla każdej klasy etykiet
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Wczytanie modelu YOLO
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# Wczytanie obrazu, na którym chcesz dokonać detekcji
image_path = 'image_test/1.jpg'  # Tutaj podaj ścieżkę do obrazu
image = cv2.imread(image_path)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Przekazanie obrazu przez sieć neuronową YOLO
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layers)

# Przetwarzanie wyjść, aby uzyskać detekcje
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:  # Prog zaufania
            center_x, center_y, w, h = map(int, detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Wykorzystanie Non-maximum suppression do usunięcia słabszych detekcji
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Rysowanie prostokątów i etykiet na obrazie
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Wyświetlenie obrazu z detekcjami
cv2.imshow('Detekcja obiektów', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
