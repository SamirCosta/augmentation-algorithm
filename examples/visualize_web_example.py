import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.visualization.visualize_web import generate_html_visualization

DATASET_DIR = "C:/Users/samir/git/augmentation/dataset_augmented"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
OUTPUT_HTML = "C:/Users/samir/git/augmentation/visualizacao_anotacoes.html"

CLASS_NAMES = ["worker"]

print("=" * 70)
print("GERANDO VISUALIZAÇÃO HTML DAS ANOTAÇÕES")
print("=" * 70)

generate_html_visualization(IMAGES_DIR, LABELS_DIR, OUTPUT_HTML, num_samples=30, class_names=CLASS_NAMES)

print("\n" + "=" * 70)
print("CONCLUÍDO!")
print("=" * 70)
print(f"\nAbra o arquivo no navegador:")
print(f"file:///{OUTPUT_HTML}")
