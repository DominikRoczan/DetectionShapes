from ultralytics import YOLO

# Ścieżka do pliku zawierającego wytrenowany model
model_path = 'runs/detect/train6/weights/best.pt'  # Zmień na właściwą ścieżkę do Twojego modelu

# Wczytaj wytrenowany model
model = YOLO(model_path)

# Ścieżka do katalogu zawierającego zdjęcia, na których chcesz przeprowadzić detekcję
images_path = 'image_test/1.jpg'  # Zmień na właściwą ścieżkę do Twojego katalogu z obrazami

# Przeprowadź detekcję na zdjęciu
results = model(images_path)

# Iteruj przez listę wyników detekcji i wywołuj metody dla każdego wyniku
for result in results:
    result.show()  # Wyświetl wyniki detekcji
    result.save()  # Zapisz wyniki detekcji