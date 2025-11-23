import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.augmenter import ConstructionSiteAugmentation


def draw_yolo_boxes(image, bboxes, class_labels, class_names=None):
    img_height, img_width = image.shape[:2]
    img_draw = image.copy()

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]

    for bbox, class_id in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox
        class_id = int(class_id)

        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)

        color = colors[class_id % len(colors)]

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"

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
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_labels, bboxes = read_yolo_labels(label_path)

    augmenter = ConstructionSiteAugmentation()
    transform = augmenter.get_transform(augmentation_level)

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    img_with_boxes = draw_yolo_boxes(image, bboxes, class_labels, class_names)
    axes[0].imshow(img_with_boxes)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

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

            aug_with_boxes = draw_yolo_boxes(aug_image, aug_bboxes, aug_labels, class_names)

            axes[i].imshow(aug_with_boxes)
            axes[i].set_title(f'Augmentation {i}', fontsize=12)
            axes[i].axis('off')

        except Exception as e:
            axes[i].text(0.5, 0.5, f'Erro: {str(e)}',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')

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
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_labels, bboxes = read_yolo_labels(label_path)

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
