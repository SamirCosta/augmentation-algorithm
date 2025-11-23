import cv2
import os
import random
from pathlib import Path

def draw_yolo_boxes(image_path, label_path, class_names=None):
    """
    Desenha as bounding boxes do formato YOLO na imagem
    """
    # Ler a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None

    height, width = image.shape[:2]

    # Cores para diferentes classes
    colors = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 255),    # Vermelho
        (255, 255, 0),  # Ciano
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarelo
    ]

    # Ler o arquivo de labels
    if not os.path.exists(label_path):
        print(f"Label não encontrado: {label_path}")
        return image

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Desenhar cada bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Converter de YOLO para coordenadas de pixel
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # Escolher cor baseada na classe
        color = colors[class_id % len(colors)]

        # Desenhar retângulo
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Adicionar label da classe
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

        # Fundo para o texto
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return image

def visualize_random_samples(images_dir, labels_dir, num_samples=10, class_names=None):
    """
    Visualiza amostras aleatórias do dataset
    """
    # Listar todas as imagens
    image_files = list(Path(images_dir).glob("*.jpg"))

    if len(image_files) == 0:
        print("Nenhuma imagem encontrada!")
        return

    print(f"Total de imagens: {len(image_files)}")

    # Selecionar amostras aleatórias
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for i, image_path in enumerate(samples):
        image_name = image_path.name
        label_path = Path(labels_dir) / image_name.replace('.jpg', '.txt')

        print(f"\n[{i+1}/{len(samples)}] Visualizando: {image_name}")

        # Desenhar boxes
        image_with_boxes = draw_yolo_boxes(str(image_path), str(label_path), class_names)

        if image_with_boxes is not None:
            # Redimensionar se a imagem for muito grande
            max_height = 800
            h, w = image_with_boxes.shape[:2]
            if h > max_height:
                scale = max_height / h
                new_w = int(w * scale)
                image_with_boxes = cv2.resize(image_with_boxes, (new_w, max_height))

            # Mostrar a imagem
            cv2.imshow(f'Anotacoes - {image_name}', image_with_boxes)

            print("Pressione qualquer tecla para a proxima imagem, 'q' para sair")
            key = cv2.waitKey(0)

            cv2.destroyAllWindows()

            if key == ord('q') or key == 27:  # 'q' ou ESC
                print("Visualizacao encerrada.")
                break

def compare_original_vs_augmented(images_dir, labels_dir, base_name, class_names=None):
    """
    Compara uma imagem original com suas versões augmentadas
    """
    # Encontrar imagem original e suas augmentadas
    original_path = Path(images_dir) / f"{base_name}.jpg"

    if not original_path.exists():
        print(f"Imagem original não encontrada: {original_path}")
        return

    # Encontrar todas as augmentadas
    augmented = list(Path(images_dir).glob(f"{base_name}_aug*.jpg"))

    print(f"\nOriginal: {base_name}.jpg")
    print(f"Augmentadas encontradas: {len(augmented)}")

    # Mostrar original
    label_path = Path(labels_dir) / f"{base_name}.txt"
    original_img = draw_yolo_boxes(str(original_path), str(label_path), class_names)

    if original_img is not None:
        cv2.imshow('Original', original_img)
        cv2.waitKey(0)

    # Mostrar cada augmentada
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

if __name__ == "__main__":
    # Configuração
    DATASET_DIR = "C:/Users/samir/git/augmentation/dataset_imgs_reais_augmented"
    IMAGES_DIR = os.path.join(DATASET_DIR, "images")
    LABELS_DIR = os.path.join(DATASET_DIR, "labels")

    # Nomes das classes (ajuste conforme seu dataset)
    CLASS_NAMES = ["worker"]  # Adicione suas classes aqui

    print("=" * 70)
    print("VISUALIZADOR DE ANOTACOES - DATASET AUGMENTADO")
    print("=" * 70)
    print("\nOpcoes:")
    print("1. Visualizar amostras aleatorias")
    print("2. Comparar original vs augmentadas")
    print("3. Visualizar uma imagem especifica")

    choice = input("\nEscolha uma opcao (1-3): ")

    if choice == "1":
        num = input("Quantas amostras visualizar? (padrao: 10): ")
        num_samples = int(num) if num.strip() else 10
        visualize_random_samples(IMAGES_DIR, LABELS_DIR, num_samples, CLASS_NAMES)

    elif choice == "2":
        # Listar algumas imagens originais
        print("\nImagens originais disponiveis (primeiras 10):")
        originals = [f.stem for f in Path(IMAGES_DIR).glob("*.jpg") if "_aug" not in f.name][:10]
        for i, name in enumerate(originals):
            print(f"{i+1}. {name}")

        idx = input("\nEscolha uma imagem (1-10): ")
        if idx.strip().isdigit() and 1 <= int(idx) <= len(originals):
            base_name = originals[int(idx) - 1]
            compare_original_vs_augmented(IMAGES_DIR, LABELS_DIR, base_name, CLASS_NAMES)
        else:
            print("Opcao invalida!")

    elif choice == "3":
        image_name = input("Nome da imagem (com .jpg): ")
        label_path = Path(LABELS_DIR) / image_name.replace('.jpg', '.txt')
        image_path = Path(IMAGES_DIR) / image_name

        if image_path.exists():
            img = draw_yolo_boxes(str(image_path), str(label_path), CLASS_NAMES)
            if img is not None:
                cv2.imshow(f'Imagem: {image_name}', img)
                print("Pressione qualquer tecla para fechar")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"Imagem nao encontrada: {image_path}")

    else:
        print("Opcao invalida!")

    print("\nVisualizacao concluida!")
