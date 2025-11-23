import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.label_fixer import fix_all_labels

LABELS_DIR = r"C:/Users/samir/git/augmentation/reais/train/labels"

fix_all_labels(LABELS_DIR, backup=True)
