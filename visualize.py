import cv2
import numpy as np
from script import ConstructionSiteAugmentation
import matplotlib.pyplot as plt
from pathlib import Path


def draw_yolo_boxes(image, bboxes, class_labels, class_names=None):
    """
    Desenha bounding boxes YOLO na imagem
    
    Args:
        image: Imagem numpy array (RGB)
        bboxes: Lista de boxes no formato YOLO [x_center, y_center, width, height]
        class_labels: Lista de IDs das classes
        class_names: Dicionário {class_id: nome}
    """
    img_height, img_width = image.shape[:2]
    img_draw = image.copy()
    
    # Cores para diferentes classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox

        # Garantir que class_id é inteiro
        class_id = int(class_id)

        # Converter de YOLO para pixel coordinates
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)

        # Escolher cor
        color = colors[class_id % len(colors)]
        
        # Desenhar retângulo
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Adicionar label
        if class_names and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        # Fundo do texto
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            img_draw,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Texto
        cv2.putText(
            img_draw,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return img_draw


def read_yolo_labels(label_path):
    """Lê arquivo de labels YOLO"""
    class_labels = []
    bboxes = []
    
    if not Path(label_path).exists():
        return class_labels, bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                class_labels.append(class_id)
                bboxes.append(bbox)
    
    return class_labels, bboxes


def visualize_augmentations(
    image_path,
    label_path,
    num_samples=6,
    augmentation_level='construction',
    class_names=None,
    save_path=None
):
    """
    Visualiza múltiplas augmentations de uma imagem
    
    Args:
        image_path: Caminho da imagem
        label_path: Caminho do label YOLO
        num_samples: Quantas augmentations mostrar
        augmentation_level: Nível de augmentation
        class_names: Dicionário com nomes das classes
        save_path: Se fornecido, salva a visualização neste caminho
    """
    # Ler imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ler labels
    class_labels, bboxes = read_yolo_labels(label_path)
    
    # Criar augmenter
    augmenter = ConstructionSiteAugmentation()
    transform = augmenter.get_transform(augmentation_level)
    
    # Configurar plot
    cols = 3
    rows = (num_samples + cols - 1) // cols  # Arredonda para cima
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    # Imagem original
    img_with_boxes = draw_yolo_boxes(image, bboxes, class_labels, class_names)
    axes[0].imshow(img_with_boxes)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Gerar augmentations
    for i in range(1, num_samples):
        try:
            if bboxes:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            else:
                transformed = transform(image=image, bboxes=[], class_labels=[])
            
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            # Desenhar boxes
            aug_with_boxes = draw_yolo_boxes(aug_image, aug_bboxes, aug_labels, class_names)
            
            axes[i].imshow(aug_with_boxes)
            axes[i].set_title(f'Augmentation {i}', fontsize=12)
            axes[i].axis('off')
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Erro: {str(e)}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Remover axes extras
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(
        f'Visualização de Augmentations - Nível: {augmentation_level}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Visualizacao salva em: {save_path}")
    
    plt.show()


def compare_augmentation_levels(
    image_path,
    label_path,
    class_names=None,
    save_path=None
):
    """
    Compara os diferentes níveis de augmentation lado a lado
    """
    # Ler imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Ler labels
    class_labels, bboxes = read_yolo_labels(label_path)
    
    # Criar augmenter
    augmenter = ConstructionSiteAugmentation()
    
    levels = ['light', 'medium', 'heavy', 'construction']
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx, level in enumerate(levels):
        transform = augmenter.get_transform(level)
        
        try:
            if bboxes:
                transformed = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            else:
                transformed = transform(image=image, bboxes=[], class_labels=[])
            
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            # Desenhar boxes
            aug_with_boxes = draw_yolo_boxes(aug_image, aug_bboxes, aug_labels, class_names)
            
            axes[idx].imshow(aug_with_boxes)
            axes[idx].set_title(f'Nível: {level.upper()}', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
            
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Erro: {str(e)}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
    
    plt.suptitle(
        'Comparação de Níveis de Augmentation',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Comparacao salva em: {save_path}")
    
    plt.show()


# ===========================================
# EXEMPLO DE USO
# ===========================================

if __name__ == "__main__":
    # Definir seus caminhos
    IMAGE_PATH = r"C:/Users/samir/git/augmentation/sinteticas/train/images/Screenshot_25_png.rf.4d96a9eef526faab4d0a817ec22a61f9.jpg"
    LABEL_PATH = r"C:/Users/samir/git/augmentation/sinteticas/train/labels/Screenshot_25_png.rf.4d96a9eef526faab4d0a817ec22a61f9.txt"
    
    # Opcional: Definir nomes das classes
    CLASS_NAMES = {
        0: 'machine',
        1: 'worker'
    }
    
    print("Gerando visualizações...")
    print("=" * 60)
    
    # Visualizar múltiplas augmentations
    print("/n1. Visualizando 6 augmentations no nível 'construction'...")
    visualize_augmentations(
        image_path=IMAGE_PATH,
        label_path=LABEL_PATH,
        num_samples=6,
        augmentation_level='construction',
        class_names=CLASS_NAMES,
        save_path='preview_augmentations.png'
    )
    
    # Comparar níveis
    print("/n2. Comparando diferentes níveis de augmentation...")
    compare_augmentation_levels(
        image_path=IMAGE_PATH,
        label_path=LABEL_PATH,
        class_names=CLASS_NAMES,
        save_path='compare_levels.png'
    )
    
    print("/n" + "=" * 60)
    print("[OK] Visualizacoes concluidas!")
    print("=" * 60)