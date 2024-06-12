import torch
from PIL import Image
import os
from img_import import image_path

# Wczytanie niestandardowego modelu
# model_path = 'runs/detect/train9/weights/best.pt'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', trust_repo=True)

image_path = image_path
img = Image.open(image_path).convert('RGB')  # Konwersja obrazu do formatu RGB

# Detekcja (przekazanie obrazu bezpośrednio do modelu)
results = model(img)

# Wyniki
results.print()  # Wydruk wyników detekcji w konsoli
# results.show()  # Pokazanie obrazu z detekcjami
path_imgsave = '/result'
results.save(path_imgsave)  # Zmień ścieżkę na właściwą lokalizację

# Sprawdzenie, czy folder istnieje, jeśli nie - utwórz go
if not os.path.exists(path_imgsave):
    os.makedirs(path_imgsave)

# Zliczanie i wyświetlanie liczby wykrytych pojazdów typu 'truck', 'car', 'bus'
detected_classes = results.xyxy[0][:, -1].numpy()  # Pobierz kolumnę z identyfikatorami klas
class_names = results.names  # Pobranie listy nazw klas

# print(class_names)

key_print = [2, 5, 7]
key_car = 2
key_bus = 5
key_truck = 71

# Liczenie wystąpień
truck_count = (detected_classes == key_truck).sum()
bus_count = (detected_classes == key_bus).sum()
car_count = (detected_classes == key_car).sum()

print(f"Detected trucks: {truck_count}")
print(f"Detected buses: {bus_count}")
print(f"Detected cars: {car_count}")
