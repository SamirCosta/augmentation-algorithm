import cv2
import os
import random
from pathlib import Path


def draw_yolo_boxes(image_path, label_path, class_names=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None

    height, width = image.shape[:2]

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    if not os.path.exists(label_path):
        print(f"Label não encontrado: {label_path}")
        return image

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        color = colors[class_id % len(colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image


def visualize_random_samples(images_dir, labels_dir, num_samples=10, class_names=None):
    image_files = list(Path(images_dir).glob("*.jpg"))

    if len(image_files) == 0:
        print("Nenhuma imagem encontrada!")
        return

    print(f"Total de imagens: {len(image_files)}")

    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for i, image_path in enumerate(samples):
        image_name = image_path.name
        label_path = Path(labels_dir) / image_name.replace('.jpg', '.txt')

        print(f"\n[{i+1}/{len(samples)}] Visualizando: {image_name}")

        image_with_boxes = draw_yolo_boxes(str(image_path), str(label_path), class_names)

        if image_with_boxes is not None:
            max_height = 800
            h, w = image_with_boxes.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                image_with_boxes = cv2.resize(image_with_boxes, (new_w, max_height))

            cv2.imshow(f'Anotacoes - {image_name}', image_with_boxes)

            print("Pressione qualquer tecla para a proxima imagem, 'q' para sair")
            key = cv2.waitKey(0)

            cv2.destroyAllWindows()

            if key == ord('q') or key == 27:
                print("Visualizacao encerrada.")
                break


def compare_original_vs_augmented(images_dir, labels_dir, base_name, class_names=None):
    original_path = Path(images_dir) / f"{base_name}.jpg"

    if not original_path.exists():
        print(f"Imagem original não encontrada: {original_path}")
        return

    augmented = list(Path(images_dir).glob(f"{base_name}_aug*.jpg"))

    print(f"\nOriginal: {base_name}.jpg")
    print(f"Augmentadas encontradas: {len(augmented)}")

    label_path = Path(labels_dir) / f"{base_name}.txt"
    original_img = draw_yolo_boxes(str(original_path), str(label_path), class_names)

    if original_img is not None:
        cv2.imshow('Original', original_img)
        cv2.waitKey(0)

    for i, aug_path in enumerate(sorted(augmented)):
        aug_label = Path(labels_dir) / aug_path.name.replace('.jpg', '.txt')
        aug_img = draw_yolo_boxes(str(aug_path), str(aug_label), class_names)

        if aug_img is not None:
            cv2.imshow(f'Augmented {i+1}', aug_img)
            print(f"Mostrando augmentada {i+1}/{len(augmented)}")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('q') or key == 27:
                break
