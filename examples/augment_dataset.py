import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.core.augmenter import DatasetAugmenter

INPUT_IMAGES = r"C:/Users/samir/git/augmentation/sinteticas/valid/images"
INPUT_LABELS = r"C:/Users/samir/git/augmentation/sinteticas/valid/labels"
OUTPUT_IMAGES = r"C:/Users/samir/git/augmentation/dataset_imgs_reais_valid_augmented/images"
OUTPUT_LABELS = r"C:/Users/samir/git/augmentation/dataset_imgs_reais_valid_augmented/labels"

print("=" * 70)
print("AUGMENTACAO DE DATASET PARA CANTEIRO DE OBRAS")
print("=" * 70)

print("/nConfiguracao:")
print(f"  Imagens de entrada: {INPUT_IMAGES}")
print(f"  Labels de entrada:  {INPUT_LABELS}")
print(f"  Imagens de saida:   {OUTPUT_IMAGES}")
print(f"  Labels de saida:    {OUTPUT_LABELS}")
print(f"  Nivel de augmentation: construction")
print(f"  Augmentations por imagem: 5")

augmenter = DatasetAugmenter(
    input_images_dir=INPUT_IMAGES,
    input_labels_dir=INPUT_LABELS,
    output_images_dir=OUTPUT_IMAGES,
    output_labels_dir=OUTPUT_LABELS,
    augmentation_level='construction'
)

print("/n" + "=" * 70)
print("Iniciando augmentacao...")
print("=" * 70)

augmenter.augment_dataset(
    num_augmentations=1,
    save_original=False
)

print("/n" + "=" * 70)
print("[OK] Processo concluido com sucesso!")
print("=" * 70)
print("/nVoce pode usar o dataset augmentado para treinar seu modelo.")
