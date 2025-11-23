import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.visualization.visualize_annotations import visualize_random_samples, compare_original_vs_augmented

DATASET_DIR = "C:/Users/samir/git/augmentation/dataset_imgs_reais_augmented"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

CLASS_NAMES = ["worker"]

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
    import cv2
    from src.visualization.visualize_annotations import draw_yolo_boxes

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
