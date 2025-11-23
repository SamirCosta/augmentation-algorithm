import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.visualization.visualize import visualize_augmentations, compare_augmentation_levels

IMAGE_PATH = r"C:/Users/samir/git/augmentation/sinteticas/train/images/Screenshot_25_png.rf.4d96a9eef526faab4d0a817ec22a61f9.jpg"
LABEL_PATH = r"C:/Users/samir/git/augmentation/sinteticas/train/labels/Screenshot_25_png.rf.4d96a9eef526faab4d0a817ec22a61f9.txt"

CLASS_NAMES = {
    0: 'machine',
    1: 'worker'
}

print("Gerando visualizações...")
print("=" * 60)

print("/n1. Visualizando 6 augmentations no nível 'construction'...")
visualize_augmentations(
    image_path=IMAGE_PATH,
    label_path=LABEL_PATH,
    num_samples=6,
    augmentation_level='construction',
    class_names=CLASS_NAMES,
    save_path='preview_augmentations.png'
)

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
